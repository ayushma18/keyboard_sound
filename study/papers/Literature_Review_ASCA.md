# Comprehensive Literature Review: Acoustic Side-Channel Attacks (ASCAs) on Keyboards

---

## 1. Introduction

Acoustic Side-Channel Attacks (ASCAs) exploit the distinct sounds produced by individual keystrokes to infer typed text. Since Asonov and Agrawal's seminal 2004 paper demonstrating ~79% keystroke recognition accuracy, the field has evolved dramatically across two decades — from basic FFT + neural network classifiers to sophisticated transformer architectures and LLM-assisted correction pipelines achieving >99% accuracy. This literature review synthesizes findings from 24 downloaded research papers (out of 60 cataloged), covering the complete ASCA lifecycle: data collection, segmentation, feature extraction, modeling, accuracy, noise handling, validation, real-world testing, and cross-domain generalization.

---

## 2. Paper-by-Paper Detailed Analysis

---

### Paper 2: Keyboard Acoustic Emanations Revisited
**Authors:** Li Zhuang, Feng Zhou, J. D. Tygar | **Year:** 2005 | **Venue:** ACM CCS 2005

| Aspect | Details |
|---|---|
| **Data Collection** | 10-minute recording of English text typing. Multiple keyboards tested (4 total, see Tables 2 & 4 in paper). Quiet and noisy environments both tested. Used standard microphone at unspecified close distance. |
| **Segmentation** | Keystrokes detected via windowed FFT energy thresholding. Push peak identified — period from keystroke start to ~100ms (push-to-release time). |
| **Features** | **Cepstrum (MFCCs)** — 32-channel Mel-Scale Filter Bank, first 16 MFCCs, 10ms windows shifted 2.5ms. Extracted from push peak over ~40ms window. Data above 12 kHz ignored. Cepstrum found **significantly superior to FFT**. |
| **Model** | Unsupervised: K-means clustering (K=50, best among K=40–55 range) → Hidden Markov Model (HMM) with bigram transition matrix from English corpus → Spelling/grammar correction → Feedback-based supervised training (linear classification and Gaussian mixtures). |
| **Accuracy** | Unsupervised HMM alone: ~60% character accuracy, ~20% word accuracy. After spelling/grammar correction: >70% character, ~50% word. After feedback training: **90–96% character accuracy** for English text, 75–90% word accuracy. Random passwords: 90% of 5-char passwords recovered in <20 attempts; 80% of 10-char passwords in <75 attempts. |
| **Noise** | Tested in both quiet and noisy environments. Performance degrades in noisy settings but remains viable. |
| **Validation** | Trained on ~10 minutes of typing, tested on separate recordings. Multiple data sets used for evaluation. |
| **Real-life Testing** | Multiple keyboards, varying environments tested. |
| **Generalization** | Cross-keyboard: accuracy drops to ~25% (top-1) when training on one keyboard and testing on another of same model. Different typists also reduce accuracy. Same keyboard + same person = best results. |
| **Key Findings** | **(1)** No labeled training data needed — unsupervised attack is viable. **(2)** Cepstrum/MFCC features dramatically outperform FFT. **(3)** Linear classification and Gaussian mixtures outperform neural networks. **(4)** Language model constraints enable high accuracy from unlabeled audio. **(5)** Framework analogous to breaking substitution ciphers. |

---

### Paper 4: Compromising Electromagnetic Emanations of Wired and Wireless Keyboards
**Authors:** Martin Vuagnoux, Sylvain Pasini | **Year:** 2009 | **Venue:** USENIX Security 2009

| Aspect | Details |
|---|---|
| **Data Collection** | Electromagnetic (not acoustic) emanations captured using USRP (Universal Software Radio Peripheral) + GNU Radio. Full-spectrum acquisition up to 2.5 GHz. 12 different keyboard models tested (PS/2, USB, wireless, laptop). 4 setups: semi-anechoic chamber, small office, adjacent office, flat in building. Distance: up to **20 meters, even through walls**. |
| **Segmentation** | Short Time Fourier Transform (Waterfall) applied to raw EM signal. Four distinct classes of compromising emanations identified: (1) Falling edges of serial cable, (2) Rising+falling edges, (3) Harmonics, (4) Matrix scan routine emanations. |
| **Model** | Signal processing + correlation-based recovery. No ML model — direct signal analysis. |
| **Accuracy** | Best attack: **95% keystroke recovery** at up to 20m through walls (PS/2). Matrix scan approach: ~2.5 bits uncertainty per keystroke across all tested keyboards. |
| **Noise** | Tested from semi-anechoic chamber (ideal) to real apartments. EM noise from other electronics present. |
| **Generalization** | All 12 keyboards vulnerable to at least one of four attacks. Each keyboard has unique fingerprint based on clock frequency inconsistencies. |
| **Key Findings** | **(1)** EM side channel complementary to acoustic. **(2)** Through-wall attacks feasible at 20m. **(3)** Modern cost-pressured keyboard designs universally vulnerable. **(4)** Keyboard fingerprinting possible even with same-model devices. |

---

### Paper 5: Acoustic Side-Channel Attacks on Printers
**Authors:** Backes, Dürmuth, Gerling, Pinkal, Sporleder | **Year:** 2010 | **Venue:** USENIX Security 2010

| Aspect | Details |
|---|---|
| **Data Collection** | Dot-matrix printer (Epson LQ-300+II). Microphone at **10cm distance** from printer. Supervised learning: words from dictionary printed and recorded. Dictionary of ~1,400 words (1,000 most common English + document-specific words). |
| **Features** | Sub-band decomposition emphasizing high frequencies (>20kHz important for printers, unlike keyboards). Linear frequency spreading (not logarithmic). Smoothing for noise robustness. Word-based approach (not letter-based) due to acoustic blurring across adjacent letters. |
| **Model** | HMMs with 3-gram word sequences for post-processing. Delayed computation of transition matrix to manage memory. Feedback-based incremental learning. |
| **Accuracy** | Up to **72% word recovery** (general English). Up to **95%** with domain-specific corpus (living-will declarations). |
| **Real-life Testing** | **In-field attack at a doctor's practice** during rush hour — recovered medical prescriptions. Observer-blind setup with chatting patients as background noise. |
| **Key Findings** | **(1)** Acoustic side channels extend to printers, not just keyboards. **(2)** Domain-specific language models dramatically boost accuracy. **(3)** Simple countermeasures (acoustic shielding, distance) suffice. **(4)** Dot-matrix printers still used by 60% of German doctors, 30% of banks. |

---

### Paper 15: A Sound for a Sound — Mitigating Acoustic Side Channel Attacks
**Authors:** S. Abhishek Anand, Nitesh Saxena | **Year:** 2016 | **Venue:** FC 2016

| Aspect | Details |
|---|---|
| **Data Collection** | Keystroke sounds for all 26 alphabetical keys recorded. Sampling frequency: **44.1 kHz**. 20 samples per key. Two typing styles tested: straw man (same finger, same angle) and hunt-and-peck (same finger, varying angle). 6-character random passwords. |
| **Segmentation** | FFT coefficient calculated with window size of 441 samples. Sum of coefficients between 0.4–22 kHz. Threshold-based peak detection. Push region: ~20ms. Release region: ~10ms (window 88 samples). |
| **Features** | Time-frequency classification (Halevi & Saxena method): combines cross-correlation value of two signals + distance between FFT features in Euclidean plane. Only push region used (17% accuracy) vs. push+release (12% accuracy). |
| **Model** | Time-frequency classifier. |
| **Accuracy** | Per-character detection: **66% average** for 6-character random passwords (from 20 repeated recordings). |
| **Defense Results** | White noise alone: attackers can filter it out — **insufficient**. **Fake keystroke sounds** overlaid on real sounds: effectively cloaks the side channel. Usability: users can still input passwords normally with masking sounds playing. |
| **Key Findings** | **(1)** First systematic study of ASCA countermeasures. **(2)** White noise alone is NOT enough. **(3)** Fake keystroke masking is the most effective defense. **(4)** Typing style significantly affects attack accuracy — hunt-and-peck is harder to attack than straw man. |

---

### Paper 17: Don't Skype & Type! — Acoustic Eavesdropping in Voice-over-IP
**Authors:** Compagno, Conti, Lain, Tsudik | **Year:** 2017 | **Venue:** ACM ASIACCS 2017

| Aspect | Details |
|---|---|
| **Data Collection** | 5 distinct users. Each user pressed A–Z sequentially, 10 times, using: (a) hunt-and-peck (single right index finger), (b) touch typing (all fingers). 6 laptops: 2x MacBook Pro 13" (2014), 2x Lenovo ThinkPad E540, 2x Toshiba Tecra M2. Recorded via **Skype VoIP** (mono audio, compressed by codecs). |
| **Segmentation** | Amplitude normalized to RMS=1. FFT coefficients summed over 10ms windows → energy threshold → extract 100ms keystroke waveform. |
| **Features** | **MFCCs** — 10ms sliding window, 2.5ms step, 32 filters in mel-scale filterbank, first 32 MFCC coefficients. MFCC outperformed FFT coefficients (90.61% vs 86.30%) and cepstral coefficients (51%). |
| **Model** | **Logistic Regression (LR)** for key classification — outperformed LDA, SVM, Random Forest, k-NN. **k-NN (k=10)** for target-device (laptop model) classification. 10-fold cross-validation. |
| **Accuracy** | Complete Profiling: **top-1: ~90%, top-5: 91.7%**. User Profiling (same device model, different user): lower but significant. Model Profiling (no knowledge of victim): **top-5: 41.89%**. Device classification: accurate enough to identify laptop model from keystroke sounds alone. |
| **Noise** | Robust to VoIP issues: bandwidth fluctuations, voice overlapping keystrokes. Performance maintained even with human speech over keystroke sounds. |
| **Real-life Testing** | Tested with Google Hangouts alongside Skype — similar vulnerability indicated. |
| **Generalization** | Cross-user: reduced but still dangerous. Cross-device (same model): moderate accuracy. Cross-device (different model): requires device identification step first. |
| **Key Findings** | **(1)** First VoIP-based remote ASCA — no physical proximity needed. **(2)** Skype faithfully transmits keystroke sounds even with codec compression. **(3)** Attacker only needs to be on a call with victim. **(4)** Device model identification from keyboard sounds is feasible. **(5)** Any VoIP call is a potential attack vector. |

---

### Paper 20: KeyDrown — Eliminating Software-Based Keystroke Timing Side-Channel Attacks
**Authors:** Schwarz, Lipp, Gruss, Weiser, Maurice, Spreitzer, Mangard | **Year:** 2018 | **Venue:** NDSS 2018

| Aspect | Details |
|---|---|
| **Focus** | Defense mechanism — injects fake keystrokes at kernel level to prevent interrupt-based and cache-based keystroke timing attacks. |
| **Method** | Propagates all keystrokes (including fakes) through the shared library stack. Attackers cannot distinguish real from fake at any level of observation. |
| **Key Findings** | **(1)** Fake keystroke injection principle directly applicable to acoustic defenses. **(2)** Kernel-level defense prevents timing side-channel attacks completely. **(3)** Negligible performance overhead. |

---

### Paper 24: Hearing Your Touch — A New Acoustic Side Channel on Smartphones
**Authors:** Shumailov, Simon, Yan, Anderson | **Year:** 2019 | **Venue:** arXiv:1903.11137

| Aspect | Details |
|---|---|
| **Data Collection** | **45 participants** in real-world environments (common room, reading room, library — all with ambient noise). Devices: 2x LG Nexus 5 phones, 1x Nexus 9 tablet. Standard **44.1 kHz** sampling. 4 experiments: (1) 10 users × 9 digits × 10 times, (2) 10 users × 200 unique 4-digit PINs, (3) letter typing, (4) 5-char words from NPS chat corpus. |
| **Segmentation** | Tap detection via high-frequency burst analysis: initial 60 samples at 8000–8500 Hz, followed by 4000–4400 Hz, then 60–70 Hz for ~1500 samples. First **128 samples** contain most information (unlike physical keyboards where release is important). Band-pass Butterworth filter (1300–1700 Hz) for TDoA calculation. |
| **Features** | **Quefrency (Cepstrum)** — NOT MFCC. Raw quefrency on first 128 samples. MFCC found unsuitable for touchscreen taps (designed for human ear, not tapping sounds). Three feature sets: Top mic, Bottom mic, Top+Bottom+delay+difference quefrency. Time Difference of Arrival (TDoA) between two microphones. |
| **Model** | **Linear Discriminant Analysis (LDA)** based on SVD. |
| **Accuracy** | PIN recovery: **61% of 200 4-digit PINs within 20 attempts** (tablet). Smartphone: 9 words of 7–13 letters recovered in 50 attempts. Macro F1 score used as primary metric. |
| **Noise** | Real-world environments with coffee machines, conversations, laptop clicks, open windows. No controlled noise — naturalistic conditions. |
| **Generalization** | Model trained offline — no need to train with victim's data. Only requires same device model. |
| **Key Findings** | **(1)** First acoustic side-channel attack on virtual (touchscreen) keyboards. **(2)** MFCCs are NOT optimal for touchscreen — quefrency is better. **(3)** Most tap information is in first 128 samples (contrast: physical keyboards use release peak). **(4)** Microphone permission enables stealthy attacks via malicious apps. **(5)** TrustZone isolation alone insufficient for protecting input. |

---

### Paper 28: Behavioral Acoustic Emanations — Attack and Verification of PIN Entry
**Authors:** Panda, Liu, Hancke, Qureshi | **Year:** 2020 | **Venue:** MDPI Sensors

| Aspect | Details |
|---|---|
| **Data Collection** | PIN entry devices (PEDs) — ATM/POS terminal keypads. 4–6 digit random PINs. Inter-keystroke timing intervals extracted from acoustic emanations. |
| **Model** | Machine learning models trained on timing features. Also proposed user verification model based on behavioral acoustic fingerprints. |
| **Accuracy** | Attack: **60% PIN key recovery**. Verification (user identification): **88% accuracy**. |
| **Key Findings** | **(1)** Inter-keystroke timing alone reveals PIN information. **(2)** Same behavioral acoustics can be used defensively for user verification. **(3)** Dual-use: attack and defense from same signal. |

---

### Paper 29: Hey Alexa, What Did I Just Type?
**Authors:** Zarandy, Shumailov, Anderson | **Year:** 2020 | **Venue:** arXiv:2012.00687

| Aspect | Details |
|---|---|
| **Focus** | Explored using voice assistants (smart speakers) as attack vectors. Always-listening devices like Amazon Echo can capture keystroke sounds from nearby keyboards. |
| **Key Findings** | **(1)** Smart speakers are a new attack vector for ASCAs. **(2)** Always-on microphones in smart home devices create persistent eavesdropping risk. |

---

### Paper 32: Zoom on the Keystrokes — Exploiting Video Calls for Keystroke Inference
**Authors:** Sabra, Maiti, Jadliwala | **Year:** 2021 | **Venue:** NDSS 2021

| Aspect | Details |
|---|---|
| **Focus** | **Video-based** (not acoustic) keystroke inference from upper-body movements visible in Zoom/Hangouts/Skype video calls. |
| **Key Findings** | **(1)** Visual side channel complements acoustic. **(2)** Shoulder/arm movements during typing are detectable. **(3)** Video calls expose dual channels (audio + video) for keystroke inference. |

---

### Paper 35: A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards
**Authors:** Joshua Harrison, Ehsan Toreini, Maryam Mehrnezhad | **Year:** 2023 | **Venue:** IEEE EuroS&PW 2023

| Aspect | Details |
|---|---|
| **Data Collection** | MacBook Pro 16-inch (2021), M1 Pro processor. **36 keys** (a–z, 0–9), 25 presses each, varying pressure and finger. **Phone recording:** iPhone 13 mini at **17cm** from keyboard on microfibre cloth (to reduce desk vibration). Stereo, **44,100 Hz**, 32 bits/sample. **Zoom recording:** MacBook's built-in microphone, single Zoom participant, noise suppression set to "low" (cannot be turned off). Output: .m4a → converted to .wav. |
| **Segmentation** | FFT-based energy thresholding. Fixed keystroke length: **14,400 samples (0.33s)**. For Zoom: adaptive threshold loop — incrementally adjusts threshold until exactly 25 keystrokes found (Algorithm 1). |
| **Features** | **Mel-spectrograms** — 64 mel bands, FFT window 1024, hop length 225 (phone) / 500 (Zoom). Produces **64×64 images**. Mel-spectrograms chosen over FFT (linear scale hides low-frequency features) and MFCC (removes potentially useful frequencies via DCT). **Data Augmentation:** SpecAugment — (1) time-shift up to 40%, (2) frequency/time masking of random 10%. |
| **Model** | **CoAtNet (Convolutional Attention Network)** — 2 depth-wise convolutional layers + 2 global relative attention layers. Output: 2D average pool → fully-connected linear layer → class probabilities. **Optimizer:** Adam. **Loss:** Cross entropy. **LR:** 5e-4 (halved from default). **Epochs:** 1100. **Batch size:** 16. Linear annealing schedule. Data split: random (not stratified). |
| **Accuracy** | **Phone: 95%** accuracy (precision 0.96, recall 0.95, F1 0.95). **Zoom: 93%** accuracy. **Highest accuracy without language model in ASCA literature at time of publication.** Top-5 accuracy not explicitly reported but confusion matrix shows most errors are adjacent keys. |
| **Noise** | Zoom's built-in noise suppression makes volume variable. Microfibre cloth reduces desk vibration (non-acoustic noise). Not tested under additional ambient noise. |
| **Validation** | Random 80/10/10 train/validation/test split. Peak validation accuracy tracked every 5 epochs. Extensive hyperparameter search (LR, epochs, split method). |
| **Real-life Testing** | Two real-world scenarios: (1) Phone near laptop (on-site attack), (2) Zoom call (remote attack). |
| **Generalization** | Single keyboard model (MacBook Pro). No cross-keyboard or cross-user testing. Authors note laptop uniformity (same model = same keyboard) increases attack surface. |
| **Key Findings** | **(1)** First use of self-attention transformer layers for keyboard ASCA. **(2)** Mel-spectrograms superior to FFT and MFCC for deep learning pipeline. **(3)** CoAtNet achieves SOTA without any language model assistance. **(4)** SpecAugment crucial for generalization. **(5)** False classifications tend to be physically adjacent keys — position on keyboard plate drives acoustic signature. **(6)** Zoom's noise suppression is a complication but not a blocker. |

---

### Paper 37: Auditory Eyesight — μs-Precision Keystroke Tracking on Unconstrained Inputs
**Authors:** Yazhou Tu et al. | **Year:** 2023 | **Venue:** USENIX Security 2023

| Aspect | Details |
|---|---|
| **Focus** | First ASCA study on **unconstrained keyboard inputs** (not limited to alphabetic keys or dictionary words). Microsecond-level signal processing based on mechanical physics of keystrokes. |
| **Key Findings** | **(1)** Threat extends beyond alphanumeric keys. **(2)** Non-line-of-sight keystroke tracking demonstrated. **(3)** Mechanical physics of individual key mechanisms creates exploitable acoustic differences. |

---

### Paper 40: Eavesdropping on VR Controller Acoustic Emanation (Heimdall)
**Authors:** Luo, Nguyen, Farooq, Sun, Yan | **Year:** 2024 | **Venue:** NDSS 2024

| Aspect | Details |
|---|---|
| **Focus** | First ASCA on **VR devices**. "Heimdall" system eavesdrops on VR controller clicking sounds to infer virtual keyboard input. |
| **Challenges** | 3D sound sources (not fixed 2D plane), variable controller placement, directional acoustic signal acquisition. |
| **Method** | Adaptive DOA-Key mapping. Directional acoustic signal acquisition. |
| **Key Findings** | **(1)** VR controllers produce exploitable click sounds. **(2)** 3D spatial audio requires different signal processing than 2D keyboard attacks. **(3)** Attack surface expanding to immersive computing environments. |

---

### Paper 41: Can Virtual Reality Protect Users from Keystroke Inference Attacks?
**Authors:** Yang, Sarwar, Hwang, Bhaskar, Zhao, Zheng | **Year:** 2024 | **Venue:** USENIX Security 2024

| Aspect | Details |
|---|---|
| **Method** | Visual attack in shared VR environments — observes avatar hand motions to infer keystrokes. |
| **Accuracy** | For 13/15 users: **86–98% of typed keys** accurately recognized. |
| **Key Findings** | **(1)** VR does NOT shield users from keystroke inference. **(2)** Avatar hand motion leaks typing information in shared virtual spaces. |

---

### Paper 42: Acoustic Side Channel Attack Based on Typing Patterns
**Authors:** Taheritajar, Rahaeimehr | **Year:** 2024 | **Venue:** arXiv:2403.08740 / CANS 2025

| Aspect | Details |
|---|---|
| **Data Collection** | IRB-approved study with **20 users**. No restrictions on keyboard type or typing pattern. Inter-keystroke time intervals used rather than raw acoustic fingerprints. |
| **Model** | Timing-based analysis — exploits inter-keystroke intervals. |
| **Accuracy** | ~**43% word detection** in realistic noisy environments. |
| **Key Findings** | **(1)** Timing patterns alone (without acoustic fingerprints) can reveal words. **(2)** Approach is keyboard-agnostic. **(3)** Discusses future LLM integration for improvement. |

---

### Paper 47: Improving ASCAs Using Transformers and LLMs
**Authors:** Park, Ayati, Cai | **Year:** 2025 | **Venue:** arXiv:2502.09782

| Aspect | Details |
|---|---|
| **Data Collection** | Same Harrison et al. dataset: MacBook Pro, 36 keys, 25 presses each. Phone and Zoom recordings. |
| **Features** | Mel-spectrograms. For VT models: images resized from 64×64 to **224×224**. Time-shift augmentation: 30–40%. SpecAugment masking. |
| **Models Tested** | **CoAtNet (tuned, "O-CoAtNet", ~24M params)**, ViT (86M), **Swin (28M)**, DeiT (86M), CLIP (87M), BEiT (86M). Optimizer: Adam (CoAtNet), AdamW (VTs). LR: 5e-4 (CoAtNet), 5e-5 (VTs). 1100 epochs. Cross entropy loss. |
| **LLM Error Correction** | GPT-4o, Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B. Few-shot prompting. Fine-tuned Llama-3.2-3B with **QLoRA** (Low-Rank Adaptation). EnglishTense dataset: 1,000 sentences (500 with digits, 500 without). Noise factors: Low/Medium/High applied to mel-spectrograms via Gaussian noise (Eq: I_noisy = I + η·N(0,1)). |
| **Accuracy** | **O-CoAtNet:** Phone: 96.45% ± 3.5% (mean), **100% max**. Zoom: 96.67% ± 2.1% (mean), **98.9% max**. This is **+5.0% (Phone)** and **+5.9% (Zoom)** over Harrison's baseline. **Swin (Direct Transform):** Best VT — comparable to CoAtNet. CLIP: significantly lower (not designed for single-modality). |
| **LLM Results** | GPT-4o boosts BLEU from ~0.07 to **~0.90** under medium noise. Fine-tuned Llama-3.2-3B achieves **98–99%** of GPT-4o's performance with **67× fewer parameters**. Progressive noise training (low → medium → high) improves robustness. |
| **Validation** | Five different seeds for statistical robustness. Mean and standard deviation reported (unlike Harrison who reported only best). Stratified and random splits compared. |
| **Key Findings** | **(1)** Vision Transformers match or exceed CoAtNet. **(2)** LLMs transform impractical noisy ASCAs into viable attacks — BLEU 0.07 → 0.90. **(3)** Fine-tuned lightweight LLMs (3B params) rival GPT-4o (200B params) for error correction. **(4)** Small datasets (25 samples/key) limit VT advantage — larger datasets expected to widen gap. |

---

### Paper 48: Making ASCAs on Noisy Keyboards Viable with LLM-Assisted "Typo" Correction (WOOT '25)
**Authors:** Ayati, Park, Cai, Botacin | **Year:** 2025 | **Venue:** USENIX WOOT '25

| Aspect | Details |
|---|---|
| **Data Collection** | Same Harrison et al. Phone and Zoom datasets. EnglishTense dataset for sentence evaluation (1,000 sentences). |
| **Noise Simulation** | Gaussian noise added to mel-spectrograms: η factors calibrated so Low/Medium/High noise produce ~95%, ~85%, ~70% baseline character accuracy. Phone noise factors: 1, 1.5, 2. Zoom: 1, 5, 6. |
| **Models** | VTs (CoAtNet, **Swin Transformer**) for classification. **GPT-4o** and **fine-tuned Llama-3.2-3B (QLoRA)** for correction. |
| **Results** | Swin achieves **new SOTA**: +5.0% Phone, +5.9% Zoom over Harrison baseline. **LLM correction: BLEU 0.07 → 0.90** under intermediate noise. Fine-tuned Llama-3.2-3B: 67× smaller than GPT-4o, achieves comparable performance. |
| **Key Findings** | **(1)** Breakthrough paper addressing **the critical real-world limitation of noise**. **(2)** Two complementary strategies: VTs capture long-range context from spectrograms, LLMs fix "typos" from noisy predictions. **(3)** Framework is open-source (EchoCrypt on GitHub). **(4)** Paradigm shift: classifiers no longer need to be perfect because LLMs reconstruct intent from noisy predictions, much as humans do. |

---

### Paper 50: Improved CoAtNet for Robust ASCA Classification
**Authors:** Hama Rawf K. et al. | **Year:** 2025 | **Venue:** International Journal of Information Security (Springer)

| Aspect | Details |
|---|---|
| **Data Collection** | **MKA (Multi-Keyboard Acoustic) datasets** — 6 platforms: HP, Lenovo, MSI, Mac, Messenger, Zoom. Raw recordings + segmented sound files + derived feature matrices. |
| **Model** | Improved CoAtNet combining convolutional layers with transformer encoders. |
| **Accuracy** | **99.8% accuracy, 99.81% precision, 99.8% recall** — **highest reported in ASCA literature**. |
| **Generalization** | First publicly reproducible **cross-platform benchmark** across 6 keyboard/recording platforms. |
| **Key Findings** | **(1)** Near-perfect accuracy achievable with improved CoAtNet on diverse datasets. **(2)** Cross-platform training/testing demonstrated. **(3)** Publicly available dataset enables reproducibility. |

---

### Paper 53: Practical Acoustic Eavesdropping on Typed Passphrases
**Authors:** Fürst, Aßmuth | **Year:** 2025 | **Venue:** arXiv:2503.16719 / CLOUD COMPUTING 2025

| Aspect | Details |
|---|---|
| **Data Collection** | Natural language passphrases (3–8 words, Diceware wordlists). Multiple participants. Keystroke segmentation: press-only vs. press+release evaluated. |
| **Features** | Compared: **Cross-correlation, MFCC, FFT**. **Cross-correlation on raw audio** confirmed as the best for unsupervised keystroke clustering, outperforming MFCC and FFT. |
| **Model** | **Unsupervised learning** — no training data required. K-Means clustering. Cross-correlation-based clustering. Hyperparameter search: K-Means (n=2049 configs), Cross-Correlation (n=2045 configs). Dictionary attack with Joint Demodulation (adapted from WiFi side-channel work). |
| **Accuracy** | Partial passphrase recovery via clustering + dictionary matching. "Faster-than-brute-force" — reduces search space significantly. |
| **Key Findings** | **(1)** Cross-correlation outperforms MFCC and FFT for unsupervised clustering. **(2)** Unsupervised methods enable real-world attacks without any prior knowledge. **(3)** Dictionary demodulation approach (from WiFi domain) successfully adapted to acoustic domain. **(4)** Passphrase recovery demonstrated despite short message length. |

---

### Paper 55: Acoustic Side-Channel Vulnerabilities — CNN Modeling Pilot Study
**Authors:** Rzemieniuk, Niewiarowski, Książek | **Year:** 2025/2026 | **Venue:** MDPI Applied Sciences

| Aspect | Details |
|---|---|
| **Data Collection** | **Keyboard 1:** Fnatic Gear Rush (Cherry MX Brown). **Keyboard 2:** Anne Pro 2 (Kailh Brown). **Microphone:** Novox NC1 condenser at **30cm** on external stand (mechanically decoupled from desk). **Sampling:** 48 kHz, WAV. 60 samples/key (26 letters), single participant, 15–20 WPM via typing.com transcription. 0.2s pre-keypress + 0.5s post-release window per keystroke. |
| **Augmentation** | 5 techniques: time shift, random gain, background noise addition, random time stretch, random pitch shift. 1,560 → **9,360 total samples** (1,560 original + 7,800 augmented). Split: 68% train / 12% validation / 20% test. |
| **Features** | **Mel-spectrograms** — 85 time frames × 128 mel-scale frequency bands. Mel-spectrograms chosen over MFCCs (DCT compression may discard keystroke-relevant frequencies). |
| **Model** | Custom **3-block CNN**: Each block = Conv2D → BatchNorm → ReLU → MaxPool → Dropout. Filters: 64 → 128 → 256. Kernel: 3×3. GlobalAveragePooling2D → Dense(128, ReLU) → Softmax(26). **Optimizer:** Adam, LR=0.001. Batch=32. Max 900 epochs with early stopping. **Post-training:** Dynamic Range Quantization (32-bit → 8-bit). |
| **Accuracy** | **Exp 1 (Baseline — same keyboard, test split):** 96.9% accuracy, loss 0.1685. Training stabilized at ~620 epochs. **Exp 2 (New clean recordings, same keyboard):** **72.55%** (816 samples). Some letters perfect (a, e, h, n, y), others struggle (j: recall 13%). **Exp 3 (Background crowd noise):** **50.52%** (861 samples) — significant drop from 72%. Letters "a", "e", "s", "w" still classified well. **Exp 4 (Different keyboard — Anne Pro 2):** Lower accuracy — model does not generalize across keyboards without fine-tuning. |
| **Noise** | Crowd noise at 50cm from mic. Accuracy drops from 72% → 50.5%. |
| **Generalization** | Cross-keyboard (Cherry MX Brown → Kailh Brown): significant accuracy drop. Adjacent keys produce similar sounds (e.g., c/x, k/l confusion). |
| **Key Findings** | **(1)** 96% test accuracy but only 72% on independently recorded samples — generalization gap. **(2)** Noise cuts accuracy nearly in half. **(3)** Mel-spectrograms confirmed superior to MFCCs. **(4)** Keyboard proximity affects acoustic similarity — adjacent keys confuse classifiers. **(5)** Quantized model (8-bit) maintains comparable accuracy. **(6)** Single-participant limitation acknowledged. |

---

## 3. Comparative Summary Tables

### 3.1 Data Collection Comparison

| Paper | Year | Keyboard | Mic Type | Distance | Sampling Rate | Participants | Keys | Samples/Key |
|---|---|---|---|---|---|---|---|---|
| Zhuang (2) | 2005 | 4 keyboards | Standard mic | Close | Not specified | 1+ | 30 | ~10 min text |
| Vuagnoux (4) | 2009 | 12 models (PS/2,USB,wireless) | EM antenna (USRP) | Up to 20m | 64 MS/s | N/A | Full | N/A |
| Backes (5) | 2010 | Dot-matrix printer | Mic | 10cm | Not specified | N/A | Word-level | ~1400 words |
| Anand (15) | 2016 | Standard keyboard | Mic | Close | 44.1 kHz | 1 | 26 (a-z) | 20 |
| Compagno (17) | 2017 | 6 laptops (3 models) | Skype (VoIP) | Remote | VoIP codec | 5 | 26 (a-z) | 10×2 styles |
| Shumailov (24) | 2019 | Touchscreen (Nexus 5, Nexus 9) | Built-in phone mics | On-device | 44.1 kHz | 45 | 9 digits + letters | 10–200 |
| Panda (28) | 2020 | PED/ATM keypad | Mic | Close | Not specified | Multiple | 10 digits | Multiple |
| Harrison (35) | 2023 | MacBook Pro 16" (2021) | iPhone 13 mini | 17cm | 44,100 Hz | 1 | 36 (a-z, 0-9) | 25 |
| Park/Ayati (47/48) | 2025 | Same as Harrison | Same as Harrison | 17cm | 44,100 Hz | 1 | 36 | 25 |
| Rawf (50) | 2025 | 6 platforms (HP,Lenovo,MSI,Mac) | Various | Various | Various | Multiple | Full | MKA dataset |
| Fürst (53) | 2025 | Standard keyboard | Mic | Close | Not specified | Multiple | 26 | Multiple |
| Rzemieniuk (55) | 2026 | 2 mech keyboards (Cherry MX, Kailh) | Novox NC1 condenser | 30cm | 48 kHz | 1 | 26 (a-z) | 60 |

### 3.2 Feature Extraction Evolution

| Era | Primary Features | Key Papers |
|---|---|---|
| 2004–2006 | FFT coefficients | Asonov (2004) |
| 2005–2017 | **MFCCs/Cepstrum** | Zhuang (2005), Compagno (2017) |
| 2019 | Quefrency (raw cepstrum, no mel weighting) | Shumailov (2019) |
| 2023–2025 | **Mel-spectrograms** (as images for DL) | Harrison (2023), Park (2025), Rzemieniuk (2026) |
| 2025 | Cross-correlation (for unsupervised) | Fürst (2025) |

### 3.3 Model Architecture Evolution

| Era | Model | Best Accuracy | Paper |
|---|---|---|---|
| 2004 | Neural Network (FFT) | ~79% | Asonov |
| 2005 | HMM + K-means + Language Model | 96% (English text) | Zhuang |
| 2017 | Logistic Regression (MFCC) | 91.7% top-5 (VoIP) | Compagno |
| 2019 | LDA (Quefrency) | 61% PINs/20 attempts | Shumailov |
| 2023 | **CoAtNet (mel-spectrogram)** | **95% phone, 93% Zoom** | Harrison |
| 2025 | **Tuned CoAtNet** | **100% phone, 98.9% Zoom** | Park |
| 2025 | **Swin Transformer** | +5.0% / +5.9% over baseline | Ayati |
| 2025 | **Improved CoAtNet (MKA)** | **99.8% cross-platform** | Rawf |
| 2025 | **VT + LLM correction** | BLEU 0.07 → 0.90 (noisy) | Ayati (WOOT) |

### 3.4 Noise Handling Across Studies

| Paper | Noise Approach | Impact |
|---|---|---|
| Zhuang (2005) | Tested quiet + noisy rooms | Performance degrades but viable |
| Anand (2016) | Tested white noise defense | White noise alone insufficient; fake keystrokes effective |
| Compagno (2017) | VoIP bandwidth fluctuations + voice overlap | Robust to VoIP-specific noise |
| Shumailov (2019) | Real-world ambient (coffee machine, conversations) | Viable in naturalistic conditions |
| Harrison (2023) | Zoom noise suppression only | Not tested with additional ambient noise |
| Ayati (2025) | **Systematic Gaussian noise injection (Low/Med/High)** | Without LLM: accuracy drops >50%. **With LLM: BLEU 0.07→0.90** |
| Rzemieniuk (2026) | Crowd noise at 50cm | 72% → 50.5% accuracy |

### 3.5 Generalization Results

| Paper | Cross-Keyboard | Cross-User | Cross-Environment |
|---|---|---|---|
| Zhuang (2005) | ~25% (same model, different unit) | Lower accuracy | Different rooms tested |
| Compagno (2017) | 6 laptops (3 models) tested | 5 users | VoIP vs. direct recording |
| Harrison (2023) | Single keyboard only | Single user only | Phone vs. Zoom |
| Rawf (2025) | **6 platforms tested — 99.8%** | Multiple users in MKA dataset | HP/Lenovo/MSI/Mac/Messenger/Zoom |
| Rzemieniuk (2026) | Cherry MX → Kailh: significant drop | Single user | Clean → noise: 72% → 50.5% |

---

## 4. Critical Analysis and Key Themes

### 4.1 Feature Extraction: The Settled Debate
The field has converged on **mel-spectrograms** as the dominant feature representation for deep learning approaches. The progression: FFT (2004) → MFCC (2005–2017) → mel-spectrograms (2023–present). Harrison (2023) provided the decisive argument: MFCCs apply a DCT that discards potentially useful frequencies, while mel-spectrograms retain all information in an image-like format optimized for CNN/transformer processing. The one exception: for unsupervised clustering, Fürst (2025) confirmed cross-correlation outperforms both MFCC and FFT.

### 4.2 The Deep Learning Inflection Point
Pre-2023 ASCA research relied on traditional ML (HMMs, SVMs, logistic regression, k-NN). Harrison's 2023 CoAtNet paper marked the inflection point where deep learning surpassed all prior methods without any language model assistance (95% vs. prior best ~91.7%). The subsequent VT/LLM integration by Park/Ayati (2025) represents the current frontier.

### 4.3 The Noise Problem — Solved?
Noise was the longstanding barrier to practical ASCAs. Accuracy routinely dropped 30–50% in noisy environments. The 2025 LLM correction paradigm (Ayati, WOOT '25) represents a potential solution: even with >50% accuracy drop from noise, LLMs recover BLEU scores to 0.90. The key insight is that **classifiers don't need to be perfect if LLMs can reconstruct intent** — analogous to how humans read text with typos.

### 4.4 Attack Surface Expansion
The attack surface has expanded enormously:
- **2004:** Dedicated microphone at close range
- **2011:** Smartphone accelerometers on same surface
- **2017:** VoIP calls (Skype) — remote, no physical proximity
- **2019:** Touchscreen virtual keyboards, smart speakers
- **2023:** Zoom calls with built-in noise suppression
- **2024:** VR controllers, laser sensors
- **2025:** Piezoelectric microphones, non-line-of-sight (screen reflections)

### 4.5 Defense Gap
Of all papers reviewed, fewer than 5 focus primarily on defense:
- **Anand (2016):** Fake keystroke masking — most effective
- **Monaco (2016):** Timing obfuscation — 20% accuracy reduction with 25ms random delay
- **Schwarz (2018):** Kernel-level fake keystroke injection — complete defense for timing attacks
- **No defense yet proven against transformer+LLM attacks**

### 4.6 Dataset Limitations
Most studies suffer from:
- **Single participant** (Harrison, Rzemieniuk)
- **Single keyboard model** (Harrison)
- **Small sample sizes** (25 samples/key is common)
- **Lab conditions** (despite claims of "practical" attacks)

The MKA dataset (Rawf, 2024) with 6 platforms is the largest publicly available resource. The SKAID dataset (Quattrone, 2025) addresses synchronized keystroke + audio collection but is recent and untested.

---

## 5. Recommendations for Future Research

1. **Larger, diverse datasets** — Multiple keyboards, users, environments, typing styles
2. **Real-time attack evaluation** — Current systems operate offline on pre-recorded audio
3. **Robust cross-keyboard generalization** — Current best (Rawf, 99.8%) uses cross-platform training; more work needed on zero-shot cross-keyboard transfer
4. **Defense research** — The attack-defense imbalance is severe; fake keystroke masking and LLM-based defenses need more exploration
5. **Integration with other modalities** — Combining acoustic + electromagnetic + video + timing for comprehensive attack/defense evaluation
6. **Standardized benchmarks** — No agreed-upon evaluation protocol exists; Harrison's dataset has become a de facto benchmark but is too small

---

## 6. Conclusion

The ASCA field has reached a remarkable maturity in 2025. The combination of mel-spectrogram features, transformer architectures (CoAtNet/Swin), and LLM-based error correction has transformed keyboard acoustic eavesdropping from a laboratory curiosity into a practical threat achieving near-perfect accuracy even in noisy conditions. The convergence of VoIP ubiquity, deep learning advances, and LLM post-processing means that any video call is now a potential attack vector. The defensive literature remains thin — fewer than 10 of 60 cataloged papers focus on countermeasures — highlighting an urgent need for systematic defense research as attack capabilities continue to improve.
