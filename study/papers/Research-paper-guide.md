# Acoustic side-channel attacks on keyboards: a comprehensive paper compendium

Acoustic side-channel attacks (ASCAs) on keyboards exploit the distinct sounds produced by different keys to infer typed text — a threat first demonstrated in 2004 that has grown dramatically more potent with modern deep learning. **This compendium catalogs 50+ research papers spanning two decades (2004–2025)**, from foundational signal-processing works to state-of-the-art transformer and LLM-based attacks achieving over 99% accuracy. The field has accelerated sharply since 2019, driven by deep learning, VoIP-based remote attack vectors exposed during the COVID-19 pandemic, and the emergence of publicly available datasets. Recent 2025 papers demonstrate that LLMs can now correct noisy predictions, making ASCAs viable even in realistic, noisy environments — a longstanding limitation.

---

## Foundational works that launched the field (2004–2009)

These papers established that keyboards leak exploitable acoustic information and defined the core attack paradigms — supervised classification, unsupervised language-model-assisted recovery, and training-free dictionary attacks.

### 1. Keyboard Acoustic Emanations
- **Authors:** Dmitri Asonov, Rakesh Agrawal
- **Year:** 2004
- **Venue:** IEEE Symposium on Security and Privacy (S&P 2004), pp. 3–11
- **Summary:** The seminal paper in this field. Demonstrated that PC keyboards, notebook keyboards, telephone pads, and ATM pads produce distinguishable sounds per key due to physical positioning on the keyboard plate. Used a neural network classifier on FFT features extracted from keystroke press peaks. Achieved **~79% top-1 and ~88% top-3 accuracy**. Tested at distances up to 15 meters with parabolic microphones. Also provided hints for designing "homophonic" keyboards resistant to acoustic attacks.
- **DOI:** 10.1109/SECPRI.2004.1301311
- **URL:** https://ieeexplore.ieee.org/document/1301311

### 2. Keyboard Acoustic Emanations Revisited
- **Authors:** Li Zhuang, Feng Zhou, J. D. Tygar
- **Year:** 2005 (CCS conference); 2009 (TISSEC journal)
- **Venue:** 12th ACM Conference on Computer and Communications Security (CCS 2005), pp. 373–382; ACM Transactions on Information and System Security (TISSEC), Vol. 13, No. 1, Article 3 (2009)
- **Summary:** Landmark paper that dramatically improved upon Asonov & Agrawal. Recovered **up to 96% of typed characters** from a 10-minute recording using *no labeled training data*. Used cepstrum features (MFCCs), Hidden Markov Models, linear classification, and feedback-based incremental learning combined with English language statistical constraints. Also demonstrated password attacks: 90% of 5-character random passwords recovered in fewer than 20 attempts.
- **DOI (journal):** 10.1145/1609956.1609959
- **DOI (conference):** 10.1145/1102120.1102169
- **URL:** https://dl.acm.org/doi/10.1145/1609956.1609959

### 3. Dictionary Attacks Using Keyboard Acoustic Emanations
- **Authors:** Yigael Berger, Avishai Wool, Arie Yeredor
- **Year:** 2006
- **Venue:** 13th ACM Conference on Computer and Communications Security (CCS 2006), pp. 245–254
- **Summary:** Training-free dictionary attack requiring no prior profiling. Reconstructed individual words of 7–13 characters from recordings under 5 seconds. Exploited the fact that physically adjacent keys produce similar acoustic signatures and used cross-correlation distance metrics. Achieved **90%+ success rate** for words of 10+ characters in the top 50 candidates, 73% overall. Runs in under 20 seconds per word.
- **DOI:** 10.1145/1180405.1180436
- **URL:** https://dl.acm.org/doi/10.1145/1180405.1180436

### 4. Compromising Electromagnetic Emanations of Wired and Wireless Keyboards
- **Authors:** Martin Vuagnoux, Sylvain Pasini
- **Year:** 2009
- **Venue:** 18th USENIX Security Symposium (USENIX Security 2009)
- **Summary:** While focused on electromagnetic (not acoustic) emanations, this is a landmark keyboard side-channel paper that intersects with acoustic research. Tested 12 keyboard models (PS/2, USB, wireless) and discovered 4 classes of compromising EM emanations. Recovered keystrokes through walls at distances up to **20 meters**.
- **URL:** https://www.usenix.org/legacy/event/sec09/tech/full_papers/vuagnoux.pdf

---

## Early expansion and new attack vectors (2010–2015)

This period saw attacks extended to smartphones as recording devices, context-free geometric approaches, and the first systematic study of typing style effects.

### 5. Acoustic Side-Channel Attacks on Printers
- **Authors:** Michael Backes, Markus Dürmuth, Sebastian Gerling, Manfred Pinkal, Caroline Sporleder
- **Year:** 2010
- **Venue:** 19th USENIX Security Symposium (USENIX Security 2010)
- **Summary:** Extended acoustic side-channel attacks to dot-matrix printers, demonstrating recovery of printed text from printer sounds using acoustic features, HMMs, and language models. Established broader principles about acoustic side channels from mechanical devices that informed keyboard research. Included an in-field demonstration at a doctor's office.
- **URL:** https://www.usenix.org/legacy/event/sec10/tech/full_papers/Backes.pdf

### 6. (sp)iPhone: Decoding Vibrations from Nearby Keyboards Using Mobile Phone Accelerometers
- **Authors:** Philip Marquardt, Arunabh Verma, Henry Carter, Patrick Traynor
- **Year:** 2011
- **Venue:** 18th ACM Conference on Computer and Communications Security (CCS 2011), pp. 551–562
- **Summary:** Demonstrated that a smartphone placed on the same surface as a keyboard can use its accelerometer (sampling at only ~100 Hz) to decode nearby keyboard vibrations with **up to 80% accuracy**. Notable because accelerometers do not require user permission, unlike microphones.
- **DOI:** 10.1145/2046707.2046771
- **URL:** https://dl.acm.org/doi/10.1145/2046707.2046771

### 7. A Closer Look at Keyboard Acoustic Emanations: Random Passwords, Typing Styles and Decoding Techniques
- **Authors:** Tzipora Halevi, Nitesh Saxena
- **Year:** 2012 (ASIACCS conference); 2015 (journal)
- **Venue:** ACM ASIACCS 2012, pp. 89–90; International Journal of Information Security, Vol. 14, No. 5, pp. 443–456 (2015)
- **Summary:** First systematic study of acoustic eavesdropping on *random* passwords where language models are inapplicable. Introduced a time-frequency decoding technique and critically examined typing style effects (hunt-and-peck vs. touch typing). Best-case **64% per-character accuracy** with matching typing style; reduced search entropy by 36–57%.
- **DOI (journal):** 10.1007/s10207-014-0264-7
- **URL:** https://link.springer.com/article/10.1007/s10207-014-0264-7

### 8. Context-Free Attacks Using Keyboard Acoustic Emanations
- **Authors:** Tong Zhu, Qiang Ma, Shanfeng Zhang, Yunhao Liu
- **Year:** 2014
- **Venue:** ACM SIGSAC Conference on Computer and Communications Security (CCS 2014), pp. 453–464
- **Summary:** Introduced context-free, geometry-based keystroke recovery using Time Difference of Arrival (TDoA) from off-the-shelf smartphones. Eliminated the need for dictionary or language context, making the attack viable against random strings including passwords. Recovered **over 72.2%** of keystrokes.
- **DOI:** 10.1145/2660267.2660296
- **URL:** https://dl.acm.org/doi/10.1145/2660267.2660296

### 9. Single-Stroke Language-Agnostic Keylogging Using Stereo-Microphones and Domain Specific Machine Learning
- **Authors:** Sashank Narain, Amirali Sanatinia, Guevara Noubir
- **Year:** 2014
- **Venue:** ACM Conference on Security and Privacy in Wireless & Mobile Networks (WiSec 2014), pp. 201–212
- **Summary:** Language-agnostic keystroke classification using stereo microphones on smartphones combined with gyroscope data. Early work combining acoustic and inertial sensor data for keystroke inference without reliance on language model constraints.
- **URL:** https://dl.acm.org/doi/10.1145/2627393.2627417

### 10. RSA Key Extraction via Low-Bandwidth Acoustic Cryptanalysis
- **Authors:** Daniel Genkin, Adi Shamir, Eran Tromer
- **Year:** 2014
- **Venue:** 34th Annual International Cryptology Conference (CRYPTO 2014), LNCS vol. 8616, pp. 444–461
- **Summary:** While not a keyboard attack, this foundational acoustic side-channel paper demonstrated RSA secret key extraction from CPU acoustic emanations during decryption. Used a mobile phone microphone placed near the target computer. Highly influential in establishing that acoustic side channels extend beyond keyboards to computational processes.
- **DOI:** 10.1007/978-3-662-44371-2_25

### 11. Snooping Keystrokes with mm-Level Audio Ranging on a Single Phone
- **Authors:** Jian Liu, Yan Wang, Gorkem Kar, Yingying Chen, Jie Yang, Marco Gruteser
- **Year:** 2015
- **Venue:** 21st ACM MobiCom (2015), pp. 142–154
- **Summary:** Showed that a *single* smartphone can acoustically snoop on keyboard keystrokes using mm-level audio ranging via TDoA from dual microphones combined with MFCC features. Training-free approach achieving **94% keystroke recovery** at 192 kHz sampling, and over 85% at standard 48 kHz. First single-device technique enabling acoustic password snooping.
- **DOI:** 10.1145/2789168.2790122
- **URL:** https://dl.acm.org/doi/abs/10.1145/2789168.2790122

### 12. Investigating the Discriminative Power of Keystroke Sound
- **Authors:** Joseph Roth, Xiaoming Liu, Arun Ross, Dimitris Metaxas
- **Year:** 2015
- **Venue:** IEEE Transactions on Information Forensics and Security (TIFS), Vol. 10, No. 2, pp. 333–345
- **Summary:** Studied the discriminative power of keystroke sound for user authentication and identification. Developed feature representations for keystroke sounds and evaluated them in authentication frameworks. Collected a comprehensive typing sounds database.
- **DOI:** 10.1109/TIFS.2014.2374424

### 13. Acoustic Attack on Keyboard Using Spectrogram and Neural Network
- **Authors:** Zdenek Martinasek, Vlastimil Clupek, Krisztina Trasy
- **Year:** 2015
- **Venue:** 38th International Conference on Telecommunications and Signal Processing (TSP 2015), IEEE, pp. 637–641
- **Summary:** Used spectrograms as direct input to a two-layer backpropagation neural network. Recorded keystrokes via a laptop's integrated microphone in an office setting — an early study of spectrogram-based classification that prefigured later deep learning approaches.

### 14. Acoustic Side Channel Attack on Enigma
- **Authors:** Ehsan Toreini, Brian Randell, Feng Hao
- **Year:** 2015
- **Venue:** Newcastle University Technical Report
- **Summary:** Historically unique investigation of a WWII Enigma machine's susceptibility to acoustic side-channel attack. Used LPC-based preprocessing and MFCC features with an ANN classifier, achieving **92.18% key recognition**. Notable for demonstrating that even mechanical encryption devices from the 1940s were vulnerable to acoustic analysis.
- **URL:** https://toreini.github.io/projects/enigma.html

---

## VoIP attacks and emerging defenses (2016–2018)

The discovery that VoIP calls faithfully transmit keystroke sounds opened a devastating remote attack vector, prompting the first systematic countermeasure research.

### 15. A Sound for a Sound: Mitigating Acoustic Side Channel Attacks on Password Keystrokes with Active Sounds
- **Authors:** S. Abhishek Anand, Nitesh Saxena
- **Year:** 2016
- **Venue:** International Conference on Financial Cryptography and Data Security (FC 2016), Springer, pp. 346–364
- **Summary:** First systematic study of countermeasures. Tested masking signals including white noise and fake keystrokes. Found that white noise alone is insufficient (attackers can filter it out), but **fake keystroke sounds** effectively cloak acoustic side-channel attacks. Evaluated both security effectiveness and usability impact.
- **URL:** https://fc16.ifca.ai/preproceedings/21_Anand.pdf

### 16. Obfuscating Keystroke Time Intervals to Avoid Identification and Impersonation
- **Authors:** John V. Monaco, Charles C. Tappert
- **Year:** 2016
- **Venue:** 9th IAPR International Conference on Biometrics (ICB 2016), IEEE
- **Summary:** Generalized the Chaum mix concept to obfuscate typing behavior. Introduced two strategies for obfuscating inter-keystroke timing. Reduced identification accuracy by 20% with a 25 ms random delay that is imperceptible to the user. Relevant to defending against acoustic attacks that exploit inter-keystroke timing.
- **arXiv:** https://arxiv.org/abs/1609.07612

### 17. Don't Skype & Type! Acoustic Eavesdropping in Voice-over-IP
- **Authors:** Alberto Compagno, Mauro Conti, Daniele Lain, Gene Tsudik
- **Year:** 2017
- **Venue:** ACM Asia Conference on Computer and Communications Security (ASIACCS 2017), pp. 703–715
- **Summary:** Groundbreaking demonstration that VoIP calls (Skype) convey enough acoustic information to reconstruct keystrokes remotely — the attacker needs only to participate in a call with the victim. With knowledge of typing style and keyboard model, achieved **top-5 accuracy of 91.7%**; without such knowledge, 41.89%. Presented at Black Hat USA 2017.
- **DOI:** 10.1145/3052973.3053005
- **arXiv:** https://arxiv.org/abs/1609.09359

### 18. SoK: Keylogging Side Channels
- **Authors:** John V. Monaco
- **Year:** 2018
- **Venue:** IEEE Symposium on Security and Privacy (S&P 2018), pp. 211–228
- **Summary:** Comprehensive systematization of knowledge covering all keylogging side channels — acoustic, electromagnetic, timing, accelerometer, optical, and more. Defined spatial vs. temporal side channels, evaluated idealized channel performance, and established a unified threat model and taxonomy. Found nontrivial information gains even with substantial measurement error.
- **DOI:** 10.1109/SP.2018.00026
- **URL:** https://ieeexplore.ieee.org/document/8418605

### 19. Keyboard Emanations in Remote Voice Calls: Password Leakage and Noise(less) Masking Defenses
- **Authors:** S. Abhishek Anand, Nitesh Saxena
- **Year:** 2018
- **Venue:** 8th ACM Conference on Data and Application Security and Privacy (CODASPY 2018), pp. 103–110
- **Summary:** Extended acoustic attack research to remote VoIP call settings and proposed software-based defense using white noise and fake keystroke masking signals virtually inserted into voice call streams without distracting the user.
- **DOI:** 10.1145/3176258.3176341
- **URL:** https://dl.acm.org/doi/10.1145/3176258.3176341

### 20. KeyDrown: Eliminating Software-Based Keystroke Timing Side-Channel Attacks
- **Authors:** Michael Schwarz, Moritz Lipp, Daniel Gruss, Samuel Weiser, Clémentine Maurice, Raphael Spreitzer, Stefan Mangard
- **Year:** 2018
- **Venue:** Network and Distributed System Security Symposium (NDSS 2018)
- **Summary:** Injects large numbers of fake keystrokes at the kernel level to prevent interrupt-based and cache-based keystroke timing attacks. Propagates all keystrokes (including fakes) through the shared library stack. The fake-keystroke injection principle is directly applicable to acoustic defenses.
- **URL:** https://www.ndss-symposium.org/wp-content/uploads/2018/02/ndss2018_04B-1_Schwarz_paper.pdf

### 21. No Training Hurdles: Fast Training-Agnostic Attacks to Infer Your Typing
- **Authors:** Song Fang, Ian Markwood, Yao Liu, Shangqing Zhao, Zhuo Lu, Haojin Zhu
- **Year:** 2018
- **Venue:** ACM SIGSAC Conference on Computer and Communications Security (CCS 2018), pp. 1747–1760
- **Summary:** Training-agnostic attack that does not require pre-collected training data specific to the target keyboard or user. Demonstrated fast, practical keystroke inference without the conventional supervised training phase.

---

## The deep learning revolution (2019–2021)

Deep neural networks — CNNs, RNNs, and their combinations — transformed ASCA from proof-of-concept attacks into practical, high-accuracy threats achieving results previously possible only with language model assistance.

### 22. Robust Keystroke Transcription from the Acoustic Side-Channel
- **Authors:** David Slater, Scott Novotney, Jessica Moore, Sean Morgan, Scott Tenaglia
- **Year:** 2019
- **Venue:** 35th Annual Computer Security Applications Conference (ACSAC 2019), pp. 776–787
- **Summary:** End-to-end deep learning system for audio-to-keystroke transcription leveraging techniques from speech recognition. Addressed the critical gap of robust keystroke detection in the presence of realistic noise and fast typing. Reduced character error rate from 36.0% to **7.41%** for known typists and from 41.3% to 15.41% for unknown typists. Collected a novel dataset of 17 users and 86,000 keystrokes.
- **DOI:** 10.1145/3359789.3359816
- **URL:** https://dl.acm.org/doi/10.1145/3359789.3359816

### 23. Keyboard Snooping from Mobile Phone Arrays with Mixed Convolutional and Recurrent Neural Networks
- **Authors:** Tyler Giallanza, Travis Siems, Elena Smith, Erik Gabrielsen, Ian Johnson, Mitchell A. Thornton, Eric C. Larson
- **Year:** 2019
- **Venue:** Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT), Vol. 3, No. 2, pp. 1–22
- **Summary:** Combined CNNs for keystroke detection with RNNs for word identification using mobile phone microphone arrays. Achieved 41.8% keystroke recognition and 27% word recognition in noisy, realistic meeting environments. Demonstrates both the feasibility and current limitations of acoustic snooping from consumer smartphones.
- **DOI:** 10.1145/3328916

### 24. Hearing Your Touch: A New Acoustic Side Channel on Smartphones
- **Authors:** Ilia Shumailov, Laurent Simon, Jeff Yan, Ross Anderson
- **Year:** 2019
- **Venue:** arXiv preprint (arXiv:1903.11137)
- **Summary:** First acoustic side-channel attack on *virtual* (touchscreen) keyboards. Sound waves from finger taps propagate through the screen surface and air; a malicious app using the device's built-in microphone infers typed text. Recovered **61% of 200 4-digit PINs** within 20 attempts. Used quefrency features instead of MFCCs, which were found unsuitable for touchscreen taps.
- **URL:** https://arxiv.org/abs/1903.11137

### 25. Skype & Type: Keyboard Eavesdropping in Voice-over-IP (Journal Version)
- **Authors:** Stefano Cecconello, Alberto Compagno, Mauro Conti, Daniele Lain, Gene Tsudik
- **Year:** 2019
- **Venue:** ACM Transactions on Privacy and Security (TOPS), Vol. 22, No. 4, pp. 1–34
- **Summary:** Extended journal version of "Don't Skype & Type!" providing comprehensive analysis of keyboard eavesdropping threats in VoIP settings with detailed evaluation and discussion of potential countermeasures including VoIP-level audio filtering.

### 26. KeyListener: Inferring Keystrokes on QWERTY Keyboard of Touch Screen through Acoustic Signals
- **Authors:** Li Lu, Jiadi Yu, Yingying Chen, Yanmin Zhu, Xiangyu Xu, Guangtao Xue, Minglu Li
- **Year:** 2019
- **Venue:** IEEE INFOCOM 2019, pp. 775–783
- **Summary:** Demonstrated indirect eavesdropping on touchscreen QWERTY keyboards using acoustic signal attenuation analysis from smartphone microphones to localize keystrokes.
- **DOI:** 10.1109/INFOCOM.2019.8737591
- **URL:** https://ieeexplore.ieee.org/document/8737591

### 27. LOL: Localization-Free Online Keystroke Tracking Using Acoustic Signals
- **Authors:** Zhongjie Qin, Jian Du, Guojun Han et al.
- **Year:** 2019
- **Venue:** Soft Computing, Vol. 23, pp. 11063–11075, Springer
- **Summary:** Localization-free system enabling transfer of prior knowledge across different keyboard positions. Achieved 99.47% keystroke detection rate, 97.27% recognition accuracy under ideal conditions, and **84.55% content recovery** despite changing locations and background noise.
- **DOI:** 10.1007/s00500-018-3659-y

### 28. Behavioral Acoustic Emanations: Attack and Verification of PIN Entry Using Keypress Sounds
- **Authors:** Sourav Panda, Yuzhe Liu, Gerhard P. Hancke, Usman M. Qureshi
- **Year:** 2020
- **Venue:** Sensors (MDPI), Vol. 20, No. 11, Article 3015
- **Summary:** Investigated side-channel attacks on PIN entry (4–6 digits) using keypress acoustic emanations. Attack model achieves **60% PIN key recovery**. Also proposed user verification (88% accuracy) from typing patterns as a secondary authentication layer.
- **DOI:** 10.3390/s20113015
- **URL:** https://www.mdpi.com/1424-8220/20/11/3015

### 29. Hey Alexa, What Did I Just Type? Decoding Smartphone Sounds with a Voice Assistant
- **Authors:** Almos Zarandy, Ilia Shumailov, Ross Anderson
- **Year:** 2020
- **Venue:** arXiv preprint (arXiv:2012.00687)
- **Summary:** Explored using a voice assistant (smart speaker) to decode keystroke sounds from nearby keyboards — an emerging attack vector where always-listening smart home devices become potential eavesdroppers.
- **URL:** https://arxiv.org/abs/2012.00687

### 30. I Know Your Keyboard Input: A Robust Keystroke Eavesdropper Based on Acoustic Signals
- **Authors:** Jianxin Bai, Bin Liu, Li Song
- **Year:** 2021
- **Venue:** 29th ACM International Conference on Multimedia (ACM MM 2021), pp. 1239–1247
- **Summary:** Proposed a robust eavesdropping scheme that estimates relative microphone-keyboard position and extracts two robust features from acoustic signals. Achieved **91.2% accuracy** with 10-fold cross-validation and 96.67% top-5 word-level accuracy. Key contribution: demonstrated **cross-user and cross-keyboard generalization**.
- **DOI:** 10.1145/3474085.3475539
- **URL:** https://dl.acm.org/doi/10.1145/3474085.3475539

### 31. An Indirect Eavesdropping Attack of Keystrokes on Touch Screen through Acoustic Sensing
- **Authors:** Jiadi Yu, Li Lu, Yingying Chen, Yanmin Zhu, Linghe Kong
- **Year:** 2021
- **Venue:** IEEE Transactions on Mobile Computing, Vol. 20, No. 2, pp. 337–351
- **Summary:** Extended journal version of KeyListener work, providing deeper analysis of acoustic-based touchscreen keystroke eavesdropping.
- **DOI:** 10.1109/TMC.2019.2947468

### 32. Zoom on the Keystrokes: Exploiting Video Calls for Keystroke Inference Attacks
- **Authors:** Mohd Sabra, Anindya Maiti, Murtuza Jadliwala
- **Year:** 2021
- **Venue:** Network and Distributed Systems Security (NDSS) Symposium 2021
- **Summary:** Designed a *video-based* (not acoustic) keystroke inference framework exploiting upper-body movements visible in Zoom/Hangouts/Skype video calls. Compared video-based performance with acoustic approaches and proposed mitigation techniques.
- **DOI:** 10.14722/ndss.2021.23063
- **arXiv:** https://arxiv.org/abs/2010.12078

---

## State-of-the-art attacks with transformers and LLMs (2022–2025)

The most recent wave of research achieves near-perfect accuracy using self-attention architectures, vision transformers, and LLMs for error correction — while also extending attacks to VR controllers, laser sensors, and piezoelectric microphones.

### 33. We Can Hear Your PIN Drop: An Acoustic Side-Channel Attack on ATM PIN Pads (PinDrop)
- **Authors:** Kiran Balagani, Matteo Cardaioli, Stefano Cecconello, Mauro Conti, Gene Tsudik
- **Year:** 2022
- **Venue:** ESORICS 2022, Springer LNCS, vol. 13554, pp. 633–652
- **Summary:** Demonstrated "PinDrop" attack on commercially available ATM metal PIN pads with 58 participants entering 5,800 PINs. At 0.3 m, recovered **96% of 4-digit PINs**. At the 2 m courtesy distance, 57% of PINs recovered in 3 attempts.
- **DOI:** 10.1007/978-3-031-17140-6_31

### 34. Behavicker: Eavesdropping Computer-Usage Activities Through Acoustic Side Channel
- **Authors:** Mengqi Chen et al.
- **Year:** 2022
- **Venue:** Wireless Communications and Mobile Computing
- **Summary:** Infers high-level computer-usage activities (not just individual keystrokes) from acoustic signals of keyboard and mouse using semantics-preserving multiscale learning. Broadens the acoustic side-channel threat model beyond character-level recovery.

### 35. A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards
- **Authors:** Joshua Harrison, Ehsan Toreini, Maryam Mehrnezhad
- **Year:** 2023
- **Venue:** IEEE European Symposium on Security and Privacy Workshops (EuroS&PW 2023), pp. 270–280
- **Summary:** The most widely cited recent ASCA paper. Implemented a **CoAtNet** (Convolutional Attention Network) model on mel-spectrogram features. Achieved **95% accuracy** from smartphone recordings (iPhone 13 mini) and **93% accuracy** from Zoom recordings — state-of-the-art without any language model. Used SpecAugment for data augmentation. Tested on a MacBook Pro keyboard. First use of self-attention transformer layers for keyboard ASCAs. Widely covered in mainstream media.
- **DOI:** 10.1109/EuroSPW59978.2023.00034
- **arXiv:** https://arxiv.org/abs/2308.01074

### 36. Password-Sniffing Acoustic Keylogger Using Machine Learning
- **Authors:** Alex Akinbi, Erkan Deniz, Aras M. Ismael, Zryan N. Rashid, Abdulkadir Sengur
- **Year:** 2023
- **Venue:** SSRN preprint
- **Summary:** Developed a proof-of-concept acoustic keylogger using ConvMixer (a vision transformer-based approach) on MFCC spectrogram images. Achieved **92.44% password recognition accuracy**, surpassing pretrained CNN models ResNet18 and VGG16.
- **URL:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4431909

### 37. Auditory Eyesight: Demystifying μs-Precision Keystroke Tracking Attacks on Unconstrained Keyboard Inputs
- **Authors:** Yazhou Tu, Liqun Shan, Md Imran Hossen, Sara Rampazzi, Kevin Butler, Xiali Hei
- **Year:** 2023
- **Venue:** 32nd USENIX Security Symposium (USENIX Security 2023), pp. 175–192
- **Summary:** First acoustic side-channel study on *unconstrained* keyboard inputs (not limited to alphabetic keys or dictionary words). Developed microsecond-level signal processing considering the mechanical physics of keystrokes. Revealed threats of non-line-of-sight keystroke sound tracking.
- **URL:** https://www.usenix.org/conference/usenixsecurity23/presentation/tu

### 38. Deep Learning Enabled Keystroke Eavesdropping Attack Over Videoconferencing Platforms
- **Year:** 2023
- **Venue:** IEEE Conference (2023)
- **Summary:** Investigated keystroke eavesdropping during videoconferencing (Zoom, Teams, Slack) using deep learning. Developed an automatic context-free keystroke inference algorithm achieving **~90% accuracy** on normal laptop keyboards. Motivated by the COVID-19 pandemic shift to remote work.
- **URL:** https://ieeexplore.ieee.org/document/10225861/

### 39. KeystrokeSniffer: An Off-the-Shelf Smartphone Can Eavesdrop on Your Privacy From Anywhere
- **Authors:** Jingyi Huang, Jianxin Bai, Xuhang Zhang, Zhen Liu, Yang Feng, Jinsong Liu, Xin Sun, Meiyi Dong, Minglu Li
- **Year:** 2024
- **Venue:** IEEE Transactions on Information Forensics and Security (TIFS), Vol. 19, pp. 6840–6855
- **Summary:** Proposed a keystroke eavesdropping algorithm **robust to unknown environments and unknown victims**. Designed environment estimation and keyboard-specific augmentation to handle real-world variability. First paper to systematically address the cross-environment transfer problem.
- **DOI:** 10.1109/TIFS.2024.3424301
- **URL:** https://ieeexplore.ieee.org/document/10587502

### 40. Eavesdropping on Controller Acoustic Emanation for Keystroke Inference Attack in Virtual Reality
- **Authors:** Shiqing Luo, Anh Nguyen, Hafsa Farooq, Kun Sun, Zhisheng Yan
- **Year:** 2024
- **Venue:** Network and Distributed System Security Symposium (NDSS 2024)
- **Summary:** First acoustic side-channel attack on VR devices. The "Heimdall" system eavesdrops on VR controller clicking sounds to infer keystrokes on virtual keyboards. Overcomes unique VR challenges (3D sound sources, variable controller placement) using directional acoustic signal acquisition and adaptive DOA-Key mapping.
- **DOI:** 10.14722/ndss.2024.24100
- **URL:** https://www.ndss-symposium.org/ndss-paper/eavesdropping-on-controller-acoustic-emanation-for-keystroke-inference-attack-in-virtual-reality/

### 41. Can Virtual Reality Protect Users from Keystroke Inference Attacks?
- **Authors:** Zhuolin Yang, Zain Sarwar, Iris Hwang, Ronik Bhaskar, Ben Y. Zhao, Haitao Zheng
- **Year:** 2024
- **Venue:** 33rd USENIX Security Symposium (USENIX Security 2024), pp. 2725–2742
- **Summary:** Demonstrated that VR does *not* shield users from keystroke inference. Designed attacks in shared virtual environments where an attacker observes another user's avatar hand motions. For 13/15 tested users, accurately recognized **86%–98% of typed keys**.
- **URL:** https://www.usenix.org/conference/usenixsecurity24/presentation/yang-zhuolin

### 42. Acoustic Side Channel Attack on Keyboards Based on Typing Patterns
- **Authors:** Alireza Taheritajar, Reza Rahaeimehr
- **Year:** 2024 (arXiv); 2025 (CANS proceedings)
- **Venue:** arXiv:2403.08740; CANS 2025, Springer LNCS vol. 16351
- **Summary:** Novel method exploiting inter-keystroke time intervals rather than raw acoustic fingerprints. Achieves ~43% word detection in realistic noisy environments without restrictions on keyboard type or typing pattern. IRB-approved study with 20 users. Discusses future LLM integration.
- **DOI:** 10.1007/978-981-95-4434-9_26
- **arXiv:** https://arxiv.org/abs/2403.08740

### 43. Multi-Keyboard Acoustic (MKA) Datasets
- **Authors:** Karwan M. H. Rawf et al.
- **Year:** 2024
- **Venue:** Data in Brief, Vol. 57, Article 110949
- **Summary:** Largest publicly available dataset for ASCA research, comprising recordings from **6 platforms** (HP, Lenovo, MSI, Mac, Messenger, Zoom). Includes raw recordings, segmented sound files, and derived feature matrices. Data collected with both hands/10 fingers and segmented using the Praat tool.
- **DOI:** 10.1016/j.dib.2024.110949
- **URL:** https://www.sciencedirect.com/science/article/pii/S2352340924009090

### 44. Keystroke Transcription from Acoustic Emanations Using Continuous Wavelet Transform
- **Authors:** Ozkan, A., Kilic, B.G., Acarturk, C.
- **Year:** 2024
- **Venue:** Machine Learning for Cyber Security (ML4CS 2023), Springer LNCS, Vol. 14541
- **Summary:** Proposes continuous wavelet transform (CWT) as an alternative to MFCC/FFT for feature extraction from keystroke sounds, with potential for improved noise robustness.
- **DOI:** 10.1007/978-981-97-2458-1_1

### 45. RefleXnoop: Passwords Snooping on NLoS Laptops Leveraging Screen-Induced Sound Reflection
- **Year:** 2024
- **Venue:** ACM CCS 2024
- **Summary:** Combines passive acoustic eavesdropping with active ultrasound probing, exploiting laptop screen reflections for enhanced sound diversity. Uses neural models for feature-to-key translation. Targets **non-line-of-sight** scenarios where the attacker cannot directly observe the keyboard.
- **URL:** https://dl.acm.org/doi/10.1145/3658644.3670341

### 46. Acoustic Side Channel Attack on Keyboard (Dell KB216)
- **Year:** 2024
- **Venue:** ResearchGate preprint
- **Summary:** Attacks a Dell Wired Keyboard KB216. Extracts features from mel-spectrograms using ResNet18. SVMs identify whitespace keystrokes with **90% accuracy** and individual characters with 50% accuracy using only 20 data points per character.
- **URL:** https://www.researchgate.net/publication/382264491

### 47. Improving Acoustic Side-Channel Attacks on Keyboards Using Transformers and Large Language Models
- **Authors:** Jin Hyun Park, Seyyed Ali Ayati, Yichen Cai
- **Year:** 2025
- **Venue:** arXiv preprint (arXiv:2502.09782)
- **Summary:** Extends Harrison et al.'s framework with Vision Transformers and LLMs. CoAtNet achieves **5.0% improvement** on phone data and 5.9% on Zoom data over previous benchmarks. LLMs perform contextual error correction. Fine-tuned lightweight LLMs with LoRA (67× fewer parameters) match heavyweight model performance.
- **URL:** https://arxiv.org/abs/2502.09782

### 48. Making Acoustic Side-Channel Attacks on Noisy Keyboards Viable with LLM-Assisted Spectrograms' "Typo" Correction
- **Authors:** Seyyed Ali Ayati, Jin Hyun Park, Yichen Cai, Marcus Botacin, Alyssa Milburn, Jiska Classen
- **Year:** 2025
- **Venue:** 19th USENIX Conference on Offensive Technologies (WOOT '25), pp. 87–101
- **Summary:** Breakthrough paper addressing the critical real-world limitation of noise. Vision Transformers achieve SOTA keystroke classification; LLMs (GPT-4o or fine-tuned Llama-3.2-3B via QLoRA) correct "typos" from noisy predictions. **BLEU scores boosted from ~0.07 to ~0.90** under intermediate noise levels. The fine-tuned model is 67× smaller than GPT-4o with comparable performance.
- **arXiv:** https://arxiv.org/abs/2504.11622
- **URL:** https://www.usenix.org/system/files/woot25-ayati.pdf

### 49. A New Pipeline for Snooping Keystroke Based on Deep Learning Algorithm
- **Authors:** M. Spata, V. Maria Russo, A. Ortis, S. Battiato
- **Year:** 2025
- **Venue:** IEEE Access, Vol. 13, pp. 24498–24514
- **Summary:** Novel pipeline using wavelet transforms for audio analysis and a **Temporal Convolutional Network (TCN)** for classification. Dynamic audio analysis splits waves based on signal peaks, enabling attacks without knowing the exact number of keystrokes. Peak accuracy of **98.3%**.
- **DOI:** 10.1109/ACCESS.2025.3536877
- **URL:** https://ieeexplore.ieee.org/document/10858134/

### 50. Improved CoAtNet for Robust Acoustic Side-Channel Attack Classification on Keyboards
- **Authors:** Hama Rawf K. et al.
- **Year:** 2025
- **Venue:** International Journal of Information Security (Springer)
- **Summary:** Improved CoAtNet combining convolutional layers with transformer encoders, trained on MKA datasets across 6 platforms. Achieves **99.8% accuracy, 99.81% precision, 99.8% recall** — the highest reported figures in ASCA literature. First publicly reproducible cross-platform benchmark.
- **DOI:** 10.1007/s10207-025-01194-x
- **URL:** https://link.springer.com/article/10.1007/s10207-025-01194-x

### 51. LaserKey: Eavesdropping Keyboard Typing Leveraging Vibrational Emanations via Laser Sensing
- **Year:** 2025
- **Venue:** IEEE Transactions on Mobile Computing
- **Summary:** Novel approach using laser sensors to capture subtle vibrations on laptop screens induced by keystrokes. Deep learning model uses MFCC, TDoA, and amplitude features. Achieves **92.2% single-key recognition** and 3% character error rate for word-level recognition. Introduces meta-learning-based domain generalization.
- **DOI:** 10.1109/TMC.2025.3529919
- **URL:** https://ieeexplore.ieee.org/document/10843854/

### 52. Keystroke Estimation via Piezoelectric Acoustic Sensing
- **Authors:** Hayata Tsunoda, Genta Irie
- **Year:** 2025
- **Venue:** IEEE Access, Vol. 13, pp. 203851–203864
- **Summary:** Deep learning-based attack using piezoelectric (contact-type) microphones. Proposed a loss function incorporating keyboard layout geometry to improve classification. Highlighted the emerging risk from compact, easily concealable piezoelectric sensors.
- **DOI:** 10.1109/ACCESS.2025.3639162
- **URL:** https://ieeexplore.ieee.org/document/11271618

### 53. Practical Acoustic Eavesdropping on Typed Passphrases
- **Authors:** Darren Fürst, Andreas Aßmuth
- **Year:** 2025
- **Venue:** arXiv:2503.16719; CLOUD COMPUTING 2025
- **Summary:** Exploits keyboard acoustic emanations to infer typed natural language passphrases via **unsupervised learning** with no training data. Confirms cross-correlation outperforms MFCC and FFT for keystroke clustering. Demonstrates partial passphrase recovery through clustering combined with dictionary attack.
- **URL:** https://arxiv.org/abs/2503.16719

### 54. Hear to Reveal (HERO): Stealing Keystroke Content from Keyboard Acoustic Side-Channel
- **Authors:** Zhiquan He, Zhihai Yang, Zicheng Cui, Yan Feng, Pinghui Wang, Zhiquan Liu
- **Year:** 2025
- **Venue:** SSRN preprint (under review)
- **Summary:** Self-supervised approach achieving **97.5% accuracy** in word input tasks and 93% in numeric passwords. Under leave-one-out protocol with 10 participants, achieves 90.5% average accuracy — strong cross-user generalization across mainstream keyboards.
- **URL:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5943496

### 55. Acoustic Side-Channel Vulnerabilities in Keyboard Input Explored Through CNN Modeling: A Pilot Study
- **Year:** 2025/2026
- **Venue:** Applied Sciences (MDPI), Vol. 16, No. 2, Article 563
- **Summary:** CNN-based pilot study achieving 96% accuracy on test data and 72% on independently recorded samples. Tests under controlled lab, noisy environments, different keyboard models, and after model quantization. Compares mel-spectrograms' superiority over MFCCs for this task.
- **URL:** https://www.mdpi.com/2076-3417/16/2/563

### 56. A Prototype for Generating Random Key Sounds to Prevent Keyboard Acoustic Side-Channel Attacks
- **Year:** 2024
- **Venue:** IEEE Conference (IEEE Xplore)
- **Summary:** Defensive work proposing a random key sound generation system with three security modes to mitigate keyboard acoustic side-channel attacks.
- **URL:** https://ieeexplore.ieee.org/document/10608505

### 57. SKAID: A Realistic Dataset for Acoustic Side-Channel Attacks with Synchronized Keyboard Audio and Keystroke Logs
- **Authors:** Benjamin Quattrone, Youakim Badr
- **Year:** 2025
- **Venue:** Zenodo (Pennsylvania State University)
- **Summary:** Introduces methodology for synchronized keystroke and acoustic data collection in naturalistic conditions, capturing keystroke-level logs, raw audio, transcribed text, and user demographics. Addresses the critical lack of realistic, publicly available datasets for ASCA research.
- **URL:** https://zenodo.org/records/17282184

---

## Survey papers synthesizing the field

### 58. A Survey on Acoustic Side Channel Attacks on Keyboards
- **Authors:** Alireza Taheritajar, Zahra Mahmoudpour Harris, Reza Rahaeimehr
- **Year:** 2023/2024
- **Venue:** arXiv:2309.11012; published in ICICS 2024, Springer LNCS vol. 15056
- **Summary:** Comprehensive survey reviewing 170+ papers. Categorizes attacks by methodology (timing-based, geometry-based, frequency-based), threat model, keyboard type, and recording medium. Covers both offense and defense strategies.
- **DOI:** 10.1007/978-981-97-8798-2_6
- **arXiv:** https://arxiv.org/abs/2309.11012

### 59. A Survey on Acoustic Side-Channel Attacks: An Artificial Intelligence Perspective
- **Year:** 2025
- **Venue:** MDPI Information, Vol. 6, No. 1, Article 6
- **Summary:** Systematic review of January 2020–February 2025 research. Categorizes methods into three levels: individual keystroke, short text (words/phrases), and long-text reconstruction. Identifies trends (TCNs, GANs achieving up to 98.3% accuracy) and remaining gaps (lack of realistic datasets, poor generalization).
- **URL:** https://www.mdpi.com/2624-800X/6/1/6

---

## Open-source tools and implementations

### 60. Keytap / kbd-audio
- **Author:** Georgi Gerganov
- **Year:** 2018–2022 (ongoing)
- **Platform:** GitHub (open-source)
- **Summary:** Suite of practical acoustic keyboard eavesdropping tools. **Keytap** (v1) requires per-keyboard training using cross-correlation. **Keytap2** needs no training data, treating text recovery as a substitution cipher via n-gram frequency analysis. **Keytap3** is fully automated with improved statistics. All run in-browser via WebAssembly.
- **URL:** https://github.com/ggerganov/kbd-audio; https://keytap3.ggerganov.com/

---

## How attack accuracy has progressed over two decades

The trajectory of this field tells a clear story of escalating threat. In 2004, Asonov and Agrawal's neural network needed labeled training data per keyboard and achieved ~79% accuracy. Zhuang et al. eliminated the training requirement in 2005 using language models, reaching 96% — but only for English text. The 2014–2015 geometry-based approaches (Zhu et al., Liu et al.) made random password attacks viable at 72–94% accuracy. Harrison et al.'s 2023 CoAtNet model hit **95% accuracy without any language model**, and the improved 2025 CoAtNet reaches 99.8% across multiple platforms. Most critically, Ayati et al.'s 2025 LLM-assisted approach makes attacks viable in noisy real-world environments for the first time, boosting BLEU scores from 0.07 to 0.90.

The attack surface has simultaneously expanded from dedicated microphones at close range (2004) to smartphones, VoIP calls, smart speakers, laser sensors, piezoelectric devices, and VR environments. Defensive countermeasures — fake keystroke masking, timing obfuscation, silent keyboards, and microphone access controls — remain an active but comparatively underexplored area of research. The field's center of gravity has shifted decisively toward deep learning, with mel-spectrograms as the dominant feature representation and transformer architectures as the classification backbone.

## Conclusion

This compendium identifies **60 papers and tools** spanning the complete lifecycle of acoustic side-channel attacks on keyboards. Three insights stand out beyond the obvious accuracy improvements. First, the convergence of VoIP ubiquity and deep learning has transformed ASCAs from laboratory curiosities into practical remote threats — any video call is now a potential attack vector. Second, the 2025 integration of LLMs for error correction represents a paradigm shift: classifiers no longer need to be perfect because language models can reconstruct intent from noisy predictions, much as humans do. Third, the defensive literature remains thin relative to the attack literature — of the 60 works cataloged here, fewer than 10 focus primarily on countermeasures, suggesting an urgent need for more systematic defense research as attack capabilities continue to improve.