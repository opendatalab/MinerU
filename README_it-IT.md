<div align="center" xmlns="http://www.w3.org/1999/xhtml">
<!-- logo -->
<p align="center">
  <img src="https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docs/images/MinerU-logo.png" width="300px" style="vertical-align:middle;">
</p>

<!-- icon -->

[![stars](https://img.shields.io/github/stars/opendatalab/MinerU.svg)](https://github.com/opendatalab/MinerU)
[![forks](https://img.shields.io/github/forks/opendatalab/MinerU.svg)](https://github.com/opendatalab/MinerU)
[![open issues](https://img.shields.io/github/issues-raw/opendatalab/MinerU)](https://github.com/opendatalab/MinerU/issues)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/opendatalab/MinerU)](https://github.com/opendatalab/MinerU/issues)
[![PyPI version](https://img.shields.io/pypi/v/mineru)](https://pypi.org/project/mineru/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mineru)](https://pypi.org/project/mineru/)
[![Downloads](https://static.pepy.tech/badge/mineru)](https://pepy.tech/project/mineru)
[![Downloads](https://static.pepy.tech/badge/mineru/month)](https://pepy.tech/project/mineru)

<a href="https://trendshift.io/repositories/11174" target="_blank"><img src="https://trendshift.io/api/badge/repositories/11174" alt="opendatalab%2FMinerU | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

<!-- language -->

[English](README.md) | [简体中文](README_zh-CN.md) | [Italiano](README_it-IT.md)

<!-- hot link -->

<p align="center">
🚀<a href="https://mineru.net/?source=github">Accedi a MinerU Ora→✅ Versione Web senza installazione ✅ Client Desktop completo ✅ Accesso API immediato; Evita i problemi di deployment — ottieni tutti i formati in un click. Sviluppatori, entrate!</a>
</p>

<!-- join us -->

<p align="center">
    👋 unisciti a noi su <a href="https://discord.gg/Tdedn9GTXq" target="_blank">Discord</a> e <a href="https://mineru.net/community-portal/?aliasId=3c430f94" target="_blank">WeChat</a>
</p>

</div>


<details>
<summary>MinerU — Motore di analisi documentale ad alta precisione per workflow LLM · RAG · Agent</summary>
Converte PDF · DOCX · PPTX · XLSX · Immagini · Pagine web in Markdown / JSON strutturato · Doppio motore VLM+OCR · 109 lingue <br>
Server MCP · Integrazione nativa LangChain / Dify / FastGPT · Supporto per oltre 10 chip AI domestici

**🔍 Funzionalità di Analisi Principali**

- Supporto nativo per l'analisi di `DOCX`, `PPTX` e `XLSX`
- Formule → LaTeX · Tabelle → HTML, ricostruzione accurata del layout
- Supporto per documenti scansionati, scrittura a mano, layout multi-colonna, unione tabelle cross-pagina
- L'output segue l'ordine di lettura umano con rimozione automatica di intestazioni/piè di pagina
- Doppio motore VLM + OCR, riconoscimento OCR in 109 lingue

**🔌 Integrazione**

| Caso d'uso | Soluzione |
|------------|----------|
| Strumenti AI per il codice | Server MCP — Cursor · Claude Desktop · Windsurf |
| Framework RAG | LangChain · LlamaIndex · RAGFlow · RAG-Anything · Flowise · Dify · FastGPT |
| Sviluppo | Python / Go / TypeScript SDK · CLI · REST API · Docker |
| Senza codice | mineru.net online · Gradio WebUI · Client desktop |

**🖥️ Deployment (Privato · Completamente Offline)**

| Backend di inferenza | Ideale per |
|---------------------|------------|
| pipeline            | Veloce e stabile, senza allucinazioni, funziona su CPU o GPU |
| vlm-engine          | Alta precisione, supporta ecosistema vLLM / LMDeploy / mlx |
| hybrid-engine       | Alta precisione, estrazione testo nativo, basse allucinazioni |

Chip AI domestici: Ascend · Cambricon · Enflame · MetaX · Moore Threads · Kunlunxin · Iluvatar · Hygon · Biren · T-Head
</details>

# Registro delle Modifiche

- 2026/04/18 3.1.0 Rilasciata

  Questa release si concentra su **apertura delle licenze, precisione di analisi e supporto nativo multiformato**. Gli aggiornamenti principali includono:

  - Aggiornamento licenza
    - MinerU è passato ufficialmente da `AGPLv3` alla [Licenza Open Source MinerU](https://github.com/opendatalab/MinerU/blob/master/LICENSE.md), una licenza personalizzata basata su `Apache 2.0`.
    - Questo cambiamento riduce significativamente le barriere all'adozione sia per gli utenti della community che per i deployment commerciali.
  - Aggiornamento modello VLM principale
    - Il modello VLM primario è stato aggiornato a `MinerU2.5-Pro-2604-1.2B`, portando la precisione di analisi complessiva allo stato dell'arte.
    - Il nuovo modello supporta ora l'analisi di immagini e grafici, l'unione di paragrafi troncati, l'unione di tabelle cross-pagina e il riconoscimento di immagini all'interno delle tabelle.
  - Supporto nativo per tutti i formati
    - Il supporto nativo all'analisi è stato esteso a `PPTX` e `XLSX`.
    - MinerU supporta ora completamente l'analisi di immagini, `PDF`, `DOCX`, `PPTX` e `XLSX`.

> 📝 Consulta il [Registro delle Modifiche](https://opendatalab.github.io/MinerU/reference/changelog/) completo per le versioni precedenti

# MinerU

## Introduzione al Progetto

MinerU è uno strumento di analisi documentale che converte input `PDF`, immagini, `DOCX`, `PPTX` e `XLSX` in formati leggibili dalle macchine come Markdown e JSON per il successivo recupero, estrazione ed elaborazione.
MinerU è nato durante il processo di pre-addestramento di [InternLM](https://github.com/InternLM/InternLM). Ci concentriamo sulla risoluzione dei problemi di conversione dei simboli nella letteratura scientifica e speriamo di contribuire allo sviluppo tecnologico nell'era dei grandi modelli.
Rispetto ai noti prodotti commerciali, MinerU è ancora giovane. Se riscontrate problemi o i risultati non sono quelli attesi, inviate una segnalazione su [issue](https://github.com/opendatalab/MinerU/issues) e **allegate il documento o il file di esempio pertinente**.

https://github.com/user-attachments/assets/4bea02c9-6d54-4cd6-97ed-dff14340982c

## Funzionalità Principali

- Supporto per input `PDF`, immagini, `DOCX`, `PPTX` e `XLSX`.
- Rimozione di intestazioni, piè di pagina, note a piè di pagina, numeri di pagina, ecc., per garantire la coerenza semantica.
- Output del testo in ordine di lettura umano, adatto per layout a colonna singola, multi-colonna e complessi.
- Preservazione della struttura del documento originale, inclusi titoli, paragrafi, elenchi, ecc.
- Estrazione di immagini, descrizioni di immagini, tabelle, titoli di tabelle e note a piè di pagina.
- Riconoscimento e conversione automatici delle formule nel documento in formato LaTeX.
- Riconoscimento e conversione automatici delle tabelle nel documento in formato HTML.
- Rilevamento automatico di PDF scansionati e PDF con testo corrotto e attivazione della funzionalità OCR.
- OCR con supporto per il rilevamento e il riconoscimento di 109 lingue.
- Supporto per molteplici formati di output, come Markdown multimodale e NLP, JSON ordinato per ordine di lettura e formati intermedi ricchi.
- Supporto per vari risultati di visualizzazione, tra cui visualizzazione del layout e visualizzazione degli span, per una conferma efficiente della qualità dell'output.
- CLI, FastAPI, Gradio WebUI integrati, per orchestrazione locale e deployment multi-servizio.
- Supporto per l'esecuzione in ambienti puramente CPU, e anche accelerazione GPU/MPS.
- Compatibile con piattaforme Windows, Linux e Mac.

# Avvio Rapido

L'analisi documentale è un compito difficile e complesso. In scenari come layout complessi, pagine scansionate e contenuti scritti a mano, i risultati dell'analisi potrebbero non soddisfare le aspettative. Raccomandiamo di provare prima la demo online per valutare la qualità di analisi e l'idoneità di MinerU prima di scegliere un metodo di deployment appropriato in base alle vostre esigenze.
Se avete campioni di **documenti** con risultati di analisi insoddisfacenti, sentitevi liberi di condividerli in un [issue](https://github.com/opendatalab/MinerU/issues). Continueremo a migliorare le capacità di analisi.
Se riscontrate problemi di installazione, consultate prima le <a href="#faq">FAQ</a>.

## Esperienza Online

### Applicazione web ufficiale online
La versione online ufficiale ha le stesse funzionalità del client, con un'interfaccia elegante e funzionalità avanzate, richiede l'accesso per l'uso

### Demo online basata su Gradio
Una WebUI sviluppata con Gradio, con interfaccia semplice e solo funzionalità di analisi principali, accesso non richiesto

## Deployment Locale

> [!WARNING]
> **Avviso pre-installazione — Supporto ambiente hardware e software**
>
> Per garantire la stabilità e l'affidabilità del progetto, ottimizziamo e testiamo solo per specifici ambienti hardware e software. Questo assicura che gli utenti che effettuano il deployment sulle configurazioni di sistema raccomandate otterranno le migliori prestazioni con il minor numero di problemi di compatibilità.
>
> Concentrando le risorse sull'ambiente principale, il nostro team può risolvere più efficientemente potenziali bug e sviluppare nuove funzionalità.
>
> Negli ambienti non principali, a causa della diversità delle configurazioni hardware e software e dei problemi di compatibilità delle dipendenze di terze parti, non possiamo garantire il 100% di disponibilità del progetto. Pertanto, per gli utenti che desiderano utilizzare questo progetto in ambienti non raccomandati, suggeriamo di leggere attentamente la documentazione e le FAQ. La maggior parte dei problemi ha già soluzioni corrispondenti nelle FAQ. Incoraggiamo anche il feedback della community per aiutarci ad espandere gradualmente il supporto.

### Installare MinerU

#### Installare MinerU usando pip o uv

```bash
pip install --upgrade pip
pip install uv
uv pip install -U "mineru[all]"
```

#### Installare MinerU dal codice sorgente

```bash
git clone https://github.com/opendatalab/MinerU.git
cd MinerU
uv pip install -e .[all]
```

> [!TIP]
> `mineru[all]` include tutte le funzionalità principali, compatibile con sistemi Windows / Linux / macOS, adatto alla maggior parte degli utenti.
> Se avete bisogno di specificare il framework di inferenza per il modello VLM, o intendete installare solo un client leggero su un dispositivo edge, fate riferimento alla documentazione [Guida all'Installazione dei Moduli di Estensione](https://opendatalab.github.io/MinerU/quick_start/extension_modules/).

---

#### Deploy MinerU tramite Docker
MinerU fornisce un comodo metodo di deployment Docker, che aiuta a configurare rapidamente l'ambiente e risolvere alcuni problemi di compatibilità.

> [!TIP]
> - Il deployment Docker è supportato solo su ambienti Linux e Windows con supporto WSL2;
> - Gli utenti macOS devono fare riferimento ai due metodi di installazione sopra indicati invece di usare il deployment Docker.

Potete ottenere le [Istruzioni per il Deployment Docker](https://opendatalab.github.io/MinerU/quick_start/docker_deployment/) nella documentazione.

---

### Utilizzo di MinerU

Se il vostro dispositivo soddisfa i requisiti di accelerazione GPU nella tabella sopra, potete usare un semplice comando per l'analisi documentale:

```bash
mineru -p <percorso_input> -o <percorso_output>
```

Se il vostro dispositivo non soddisfa i requisiti di accelerazione GPU, potete specificare il backend come `pipeline` per l'esecuzione in un ambiente puramente CPU:

```bash
mineru -p <percorso_input> -o <percorso_output> -b pipeline
```

`mineru` supporta attualmente input di file o directory locali `PDF`, immagini, `DOCX`, `PPTX` e `XLSX`, e può essere utilizzato per l'analisi documentale tramite CLI, API, WebUI e `mineru-router`. Per istruzioni dettagliate, consultate la [Guida all'Uso](https://opendatalab.github.io/MinerU/usage/).

# FAQ

- Se riscontrate problemi durante l'utilizzo, potete prima consultare le [FAQ](https://opendatalab.github.io/MinerU/faq/) per le soluzioni.
- Se il vostro problema rimane irrisolto, potete anche usare [DeepWiki](https://deepwiki.com/opendatalab/MinerU) per interagire con un assistente AI, che può risolvere la maggior parte dei problemi comuni.
- Se ancora non riuscite a risolvere il problema, siete benvenuti a unirvi alla nostra community tramite [Discord](https://discord.gg/Tdedn9GTXq) o [WeChat](https://mineru.net/community-portal/?aliasId=3c430f94) per discutere con altri utenti e sviluppatori.

# Grazie a Tutti i Nostri Contributori

<a href="https://github.com/opendatalab/MinerU/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=opendatalab/MinerU" />
</a>

# Informazioni sulla Licenza

Questo repository è rilasciato sotto la [Licenza Open Source MinerU](https://github.com/opendatalab/MinerU/blob/master/LICENSE.md), basata su Apache 2.0 con condizioni aggiuntive.

# Ringraziamenti

- [UniMERNet](https://github.com/opendatalab/UniMERNet)
- [TableStructureRec](https://github.com/RapidAI/TableStructureRec)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleOCR2Pytorch](https://github.com/frotms/PaddleOCR2Pytorch)
- [fast-langdetect](https://github.com/LlmKira/fast-langdetect)
- [pypdfium2](https://github.com/pypdfium2-team/pypdfium2)
- [pdftext](https://github.com/datalab-to/pdftext)
- [pdfminer.six](https://github.com/pdfminer/pdfminer.six)
- [pypdf](https://github.com/py-pdf/pypdf)
- [magika](https://github.com/google/magika)
- [vLLM](https://github.com/vllm-project/vllm)
- [LMDeploy](https://github.com/InternLM/lmdeploy)

# Citazione

```bibtex
@article{wang2026mineru2,
  title={MinerU2. 5-Pro: Pushing the Limits of Data-Centric Document Parsing at Scale},
  author={Wang, Bin and He, Tianyao and Ouyang, Linke and Wu, Fan and Zhao, Zhiyuan and Chu, Tao and Qu, Yuan and Jin, Zhenjiang and Zeng, Weijun and Miao, Ziyang and others},
  journal={arXiv preprint arXiv:2604.04771},
  year={2026}
}

@article{dong2026minerudiffusion,
  title={MinerU-Diffusion: Rethinking Document OCR as Inverse Rendering via Diffusion Decoding},
  author={Dong, Hejun and Niu, Junbo and Wang, Bin and Zeng, Weijun and Zhang, Wentao and He, Conghui},
  journal={arXiv preprint arXiv:2603.22458},
  year={2026}
}

@article{niu2025mineru2,
  title={Mineru2. 5: A decoupled vision-language model for efficient high-resolution document parsing},
  author={Niu, Junbo and Liu, Zheng and Gu, Zhuangcheng and Wang, Bin and Ouyang, Linke and Zhao, Zhiyuan and Chu, Tao and He, Tianyao and Wu, Fan and Zhang, Qintong and others},
  journal={arXiv preprint arXiv:2509.22186},
  year={2025}
}

@article{wang2024mineru,
  title={Mineru: An open-source solution for precise document content extraction},
  author={Wang, Bin and Xu, Chao and Zhao, Xiaomeng and Ouyang, Linke and Wu, Fan and Zhao, Zhiyuan and Xu, Rui and Liu, Kaiwen and Qu, Yuan and Shang, Fukai and others},
  journal={arXiv preprint arXiv:2409.18839},
  year={2024}
}

@article{he2024opendatalab,
  title={Opendatalab: Empowering general artificial intelligence with open datasets},
  author={He, Conghui and Li, Wei and Jin, Zhenjiang and Xu, Chao and Wang, Bin and Lin, Dahua},
  journal={arXiv preprint arXiv:2407.13773},
  year={2024}
}
```

# Cronologia Stelle

<a>
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=opendatalab/MinerU&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=opendatalab/MinerU&type=Date" />
   <img alt="Grafico Cronologia Stelle" src="https://api.star-history.com/svg?repos=opendatalab/MinerU&type=Date" />
 </picture>
</a>


# Link
- [MinerU-Diffusion: Rethinking Document OCR as Inverse Rendering via Diffusion Decoding](https://github.com/opendatalab/MinerU-Diffusion)
- [Easy Data Preparation with latest LLMs-based Operators and Pipelines](https://github.com/OpenDCAI/DataFlow)
- [Vis3 (Browser OSS basato su s3)](https://github.com/opendatalab/Vis3)
- [LabelU (Strumento di annotazione dati multi-modale leggero)](https://github.com/opendatalab/labelU)
- [LabelLLM (Piattaforma open-source per l'annotazione di dialoghi LLM)](https://github.com/opendatalab/LabelLLM)
- [PDF-Extract-Kit (Toolkit completo per l'estrazione di contenuti PDF ad alta qualità)](https://github.com/opendatalab/PDF-Extract-Kit)
- [OmniDocBench (Benchmark completo per l'analisi e la valutazione documentale)](https://github.com/opendatalab/OmniDocBench)
- [Magic-HTML (Strumento di estrazione per pagine web miste)](https://github.com/opendatalab/magic-html)
- [Magic-Doc (Strumento di estrazione veloce per ppt/pptx/doc/docx/pdf)](https://github.com/InternLM/magic-doc)
- [Dingo: Strumento completo di valutazione della qualità dei dati AI](https://github.com/MigoXLab/dingo)
