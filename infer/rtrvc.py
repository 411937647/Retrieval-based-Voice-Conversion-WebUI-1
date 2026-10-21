import os
import traceback
from time import time as ttime
import faiss
import numpy as np
import parselmouth
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Resample

from infer import hubert
from infer.hubert import extract_hubert_features, load_hubert_model
from i18n.i18n import I18nAuto
from tools.cuda_graph import run_cuda_graph


i18n = I18nAuto()


def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)


def read_index(index_path):
    """读取 faiss 索引并显式设置 nprobe。

    nprobe 会随索引文件持久化，而 train_index.py 建索引时写死为 1（文件名里的
    nprobe_1 即由此而来）。10 万级向量 / 2600 聚类的 IVF 索引在 nprobe=1 时
    每次只扫约 39 条向量，recall@8 只有约 0.5，检索基本失效。
    可用环境变量 RVC_INDEX_NPROBE 覆盖（默认 16）。
    """
    index = faiss.read_index(index_path)
    ivf = faiss.try_extract_index_ivf(index)
    if ivf is not None:
        want = int(os.environ.get("RVC_INDEX_NPROBE", 16))
        ivf.nprobe = max(1, min(want, ivf.nlist))
        printt(i18n("索引检索：nlist=%d, nprobe=%d"), ivf.nlist, ivf.nprobe)
    return index


def get_synthesizer(pth_path, device=torch.device("cpu")):
    from infer.module.models import (
        SynthesizerTrnMs256NSFsid,
        SynthesizerTrnMs256NSFsid_nono,
        SynthesizerTrnMs768NSFsid,
        SynthesizerTrnMs768NSFsid_nono,
    )

    cpt = torch.load(pth_path, map_location=torch.device("cpu"))
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=False)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=False)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g = net_g.float()
    net_g.eval().to(device)
    net_g.remove_weight_norm()
    return net_g, cpt


# config.device=torch.device("cpu")########强制cpu测试
# config.is_half=False########强制cpu测试
class RVC:
    def __init__(
        self,
        key,
        formant,
        pth_path,
        index_path,
        index_rate,
        config,
        last_rvc=None,
        hubert_dir=None,
    ) :
        """
        初始化
        """
        try:
            # global config
            self.config = config
            # device="cpu"########强制cpu测试
            self.device = config.device
            self.f0_up_key = key
            self.formant_shift = formant
            # 模型内嵌的说话人 id（emb_g），0 为默认音色；可用 change_sid 热切换
            self.spk_id = 0
            self.n_spk = 1
            # 清辅音保护：0=最强保护，0.5=关闭（与离线 pipeline 的 protect 语义一致）
            self.protect = 0.33
            self.f0_min = 50
            self.f0_max = 1100
            self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
            self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
            self.is_half = config.is_half
            if index_rate != 0:
                self.index = read_index(index_path)
                self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
                printt(i18n("已启用索引检索"))
            self.pth_path = pth_path
            self.index_path = index_path
            self.index_rate = index_rate
            self.cache_pitch = torch.zeros(
                1024, device=self.device, dtype=torch.long
            )
            self.cache_pitchf = torch.zeros(
                1024, device=self.device, dtype=torch.float32
            )
            self.infer_count = 0

            self.resample_kernel = {}

            # hubert 特征提取器目录：默认官方转换版；切到 assets/hubert_chinese
            # 即"中文耳朵"。目录与上次一致时直接复用已加载的模型（省显存与时间）。
            self.hubert_dir = hubert_dir or ""
            if (
                last_rvc is not None
                and getattr(last_rvc, "hubert_dir", "") == self.hubert_dir
                and hasattr(last_rvc, "model")
                and last_rvc.model is not None
            ):
                self.model = last_rvc.model
                self.hubert_norm = last_rvc.hubert_norm
            else:
                if self.hubert_dir:
                    hubert.set_hubert_path(self.hubert_dir)
                self.hubert_norm = hubert.hubert_audio_requires_normalization(
                    self.hubert_dir or None
                )
                self.model = load_hubert_model(
                    self.device, self.is_half, self.hubert_dir or None
                )

            self.net_g = None

            def set_synthesizer():
                self.net_g, cpt = get_synthesizer(self.pth_path, self.device)
                self.tgt_sr = cpt["config"][-1]
                cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
                self.n_spk = cpt["config"][-3]
                self.if_f0 = cpt.get("f0", 1)
                self.version = cpt.get("version", "v1")
                if self.is_half:
                    self.net_g = self.net_g.half()
                else:
                    self.net_g = self.net_g.float()

            if last_rvc is None or last_rvc.pth_path != self.pth_path:
                set_synthesizer()
            else:
                self.tgt_sr = last_rvc.tgt_sr
                self.if_f0 = last_rvc.if_f0
                self.version = last_rvc.version
                self.is_half = last_rvc.is_half
                self.n_spk = last_rvc.n_spk
                self.net_g = last_rvc.net_g

            if last_rvc is not None and hasattr(last_rvc, "model_rmvpe"):
                self.model_rmvpe = last_rvc.model_rmvpe
            if last_rvc is not None and hasattr(last_rvc, "model_fcpe"):
                self.model_fcpe = last_rvc.model_fcpe
        except:
            printt(traceback.format_exc())

    def change_key(self, new_key):
        self.f0_up_key = new_key

    def change_sid(self, new_sid):
        """切换模型内嵌的说话人 id。

        sid 会作为张量输入传给 net_g.infer，CUDA Graph 每次会把输入拷贝进
        静态缓冲，因此热更新不需要重新捕获图。
        """
        self.spk_id = max(0, min(int(new_sid), self.n_spk - 1))
        return self.spk_id

    def change_formant(self, new_formant):
        self.formant_shift = new_formant

    def change_protect(self, new_protect):
        """设置清辅音保护强度（0=最强，0.5=关闭）。"""
        self.protect = max(0.0, min(float(new_protect), 0.5))
        return self.protect

    def preload_index(self):
        """启动时预读索引，避免变声中把 index_rate 从 0 拖起来时卡一次。

        索引 10 万条 × 768 维，reconstruct_n 需要额外约 300MB 主机内存。
        """
        if hasattr(self, "index"):
            return False
        if not self.index_path or not os.path.isfile(self.index_path):
            return False
        self.index = read_index(self.index_path)
        self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
        printt(i18n("索引已预载入内存"))
        return True

    def change_index_rate(self, new_index_rate):
        if new_index_rate != 0 and not hasattr(self, "index"):
            self.index = read_index(self.index_path)
            self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
            printt(i18n("已启用索引检索"))
        self.index_rate = new_index_rate

    def get_f0_post(self, f0):
        if not torch.is_tensor(f0):
            f0 = torch.from_numpy(f0)
        f0 = f0.float().to(self.device).squeeze()
        f0_mel = 1127 * torch.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (
            self.f0_mel_max - self.f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = torch.round(f0_mel).long()
        return f0_coarse, f0

    def get_f0(self, x, f0_up_key, method="rmvpe"):
        if method == "rmvpe":
            return self.get_f0_rmvpe(x, f0_up_key)
        if method == "fcpe":
            return self.get_f0_fcpe(x, f0_up_key)
        if method != "pm":
            raise ValueError(f"Unsupported F0 method: {method}")
        x = x.cpu().numpy()
        p_len = x.shape[0] // 160 + 1
        f0_min = 65
        l_pad = int(np.ceil(1.5 / f0_min * 16000))
        r_pad = l_pad + 1
        s = parselmouth.Sound(np.pad(x, (l_pad, r_pad)), 16000).to_pitch_ac(
            time_step=0.01,
            voicing_threshold=0.6,
            pitch_floor=f0_min,
            pitch_ceiling=1100,
        )
        assert np.abs(s.t1 - 1.5 / f0_min) < 0.001
        f0 = s.selected_array["frequency"]
        if len(f0) < p_len:
            f0 = np.pad(f0, (0, p_len - len(f0)))
        f0 = f0[:p_len]
        uv = f0 == 0
        if np.any(~uv):
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        f0 *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0)

    def get_f0_rmvpe(self, x, f0_up_key):
        if hasattr(self, "model_rmvpe") == False:
            from infer.rmvpe import RMVPE

            printt(i18n("正在加载RMVPE模型"))
            self.model_rmvpe = RMVPE(
                "assets/rmvpe/rmvpe.pt",
                is_half=self.is_half,
                device=self.device,
            )
        f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        uv = f0 == 0
        if np.any(~uv):
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        f0 *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0)

    def get_f0_fcpe(self, x, f0_up_key):
        if hasattr(self, "model_fcpe") == False:
            from infer.fcpe import FCPEInfer

            printt("Loading fcpe model")
            self.model_fcpe = FCPEInfer(self.device)
        f0 = self.model_fcpe.infer(
            x.unsqueeze(0).float(),
            sr=16000,
            decoder_mode="local_argmax",
            threshold=0.006,
        ).squeeze().detach().cpu().numpy()
        uv = f0 == 0
        if np.any(~uv):
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        f0 *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0)

    def infer(
        self,
        input_wav,
        block_frame_16k,
        skip_head,
        return_length,
        f0method,
    ) :
        report_status = self.infer_count < 3 or self.infer_count % 100 == 0
        self.infer_count += 1
        t1 = ttime()
        with torch.no_grad():
            if self.config.is_half:
                feats = input_wav.half().view(1, -1)
            else:
                feats = input_wav.float().view(1, -1)
            padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
            feats = extract_hubert_features(
                self.model,
                feats,
                self.version,
                padding_mask=padding_mask,
                normalize=self.hubert_norm,
            )
            feats = torch.cat((feats, feats[:, -1:, :]), 1)
            # protect 开启时先留一份未做索引混合的特征，用于后面按浊音掩码混回
            feats0 = feats.clone() if self.protect < 0.5 else None
        t2 = ttime()
        try:
            if hasattr(self, "index") and self.index_rate != 0:
                npy = feats[0][skip_head // 2 :].cpu().numpy().astype("float32")
                score, ix = self.index.search(npy, k=8)
                if (ix >= 0).all():
                    # score 恰为 0（查询向量与索引内某条完全重合）时 1/score 为 inf，
                    # 归一化后整行变 NaN，且会经 sola_buffer 永久污染后续输出，
                    # 这里先做下限保护。
                    score = np.maximum(score, 1e-6)
                    weight = np.square(1 / score)
                    weight /= weight.sum(axis=1, keepdims=True)
                    npy = np.sum(
                        self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1
                    )
                    if self.config.is_half:
                        npy = npy.astype("float16")
                    feats[0][skip_head // 2 :] = (
                        torch.from_numpy(npy).unsqueeze(0).to(self.device)
                        * self.index_rate
                        + (1 - self.index_rate) * feats[0][skip_head // 2 :]
                    )
                else:
                    printt(
                        i18n("索引无效：必须使用added_xxxx.index，不能使用trained_xxxx.index")
                    )
            else:
                if report_status:
                    printt(i18n("索引检索失败或未启用"))
        except Exception:
            traceback.print_exc()
            printt(i18n("索引检索失败"))
        t3 = ttime()
        p_len = input_wav.shape[0] // 160
        factor = pow(2, self.formant_shift / 12)
        return_length2 = int(np.ceil(return_length * factor))
        if self.if_f0 == 1:
            f0_extractor_frame = block_frame_16k + 800
            if f0method == "rmvpe":
                f0_extractor_frame = 5120 * ((f0_extractor_frame - 1) // 5120 + 1) - 160
            pitch, pitchf = self.get_f0(
                input_wav[-f0_extractor_frame:],
                self.f0_up_key - self.formant_shift,
                f0method,
            )
            shift = block_frame_16k // 160
            self.cache_pitch[:-shift] = self.cache_pitch[shift:].clone()
            self.cache_pitchf[:-shift] = self.cache_pitchf[shift:].clone()
            self.cache_pitch[4 - pitch.shape[0] :] = pitch[3:-1]
            self.cache_pitchf[4 - pitch.shape[0] :] = pitchf[3:-1]
            cache_pitch = self.cache_pitch[None, -p_len:]
            cache_pitchf = self.cache_pitchf[None, -p_len:] * return_length2 / return_length
        t4 = ttime()
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        feats = feats[:, :p_len, :]
        # 清辅音保护：清辅音帧（pitchf<=0）按 protect 比例保留原始特征，
        # 避免 index 混合把 s/sh/f/t/k 抹平成"四→是"。
        # 与离线 infer/vc/pipeline.py:206-212 同一套逻辑，只是掩码取缓存后的 f0。
        if self.protect < 0.5 and feats0 is not None and self.if_f0 == 1:
            feats0 = F.interpolate(
                feats0.permute(0, 2, 1), scale_factor=2
            ).permute(0, 2, 1)
            feats0 = feats0[:, :p_len, :]
            pitchff = cache_pitchf.clone()
            pitchff[pitchff > 0] = 1
            pitchff[pitchff < 1] = self.protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)
        p_len_tensor = torch.LongTensor([p_len]).to(self.device)
        sid = torch.LongTensor([int(self.spk_id)]).to(self.device)
        skip_head_value = int(skip_head)
        return_length_value = int(return_length)
        return_length2_value = int(return_length2)
        with torch.no_grad():
            if self.if_f0 == 1:
                infered_audio = run_cuda_graph(
                    self.net_g,
                    "rvc-realtime-f0-%s-%s-%s"
                    % (skip_head_value, return_length_value, return_length2_value),
                    lambda phone, lengths, coarse, continuous, speaker: self.net_g.infer(
                        phone,
                        lengths,
                        coarse,
                        continuous,
                        speaker,
                        skip_head_value,
                        return_length_value,
                        return_length2_value,
                    )[0],
                    feats,
                    p_len_tensor,
                    cache_pitch,
                    cache_pitchf,
                    sid,
                )
            else:
                infered_audio = run_cuda_graph(
                    self.net_g,
                    "rvc-realtime-no-f0-%s-%s-%s"
                    % (skip_head_value, return_length_value, return_length2_value),
                    lambda phone, lengths, speaker: self.net_g.infer(
                        phone,
                        lengths,
                        speaker,
                        skip_head_value,
                        return_length_value,
                        return_length2_value,
                    )[0],
                    feats,
                    p_len_tensor,
                    sid,
                )
        infered_audio = infered_audio.squeeze(1).float()
        upp_res = int(np.floor(factor * self.tgt_sr // 100))
        if upp_res != self.tgt_sr // 100:
            if upp_res not in self.resample_kernel:
                self.resample_kernel[upp_res] = Resample(
                    orig_freq=upp_res,
                    new_freq=self.tgt_sr // 100,
                    dtype=torch.float32,
                ).to(self.device)
            infered_audio = self.resample_kernel[upp_res](
                infered_audio[:, : return_length * upp_res]
            )
        t5 = ttime()
        if report_status:
            printt(
                i18n("耗时：特征=%.3f秒，索引=%.3f秒，音高=%.3f秒，模型=%.3f秒"),
                t2 - t1,
                t3 - t2,
                t4 - t3,
                t5 - t4,
            )
        return infered_audio.squeeze()
