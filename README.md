# ScrapeResearch.ai
**Built with:** Python • LangChain • Mistral AI • Tavily • Streamlit.

ScrapeResearch.ai turns a research topic into a structured report through a four-stage workflow: web search, webpage extraction, report writing, and AI-based critique. It combines live search results and extracted webpage content with an LLM-powered writing pipeline, then presents the report, sources, and critique in a Streamlit interface.

<p align="center">
  <img src="images/Home%20Page.png" alt="ScrapeResearch.ai home page" width="100%" />
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python" /></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit" /></a>
  <a href="https://www.langchain.com/"><img src="https://img.shields.io/badge/Orchestration-LangChain-1C3C3C?logo=langchain&logoColor=white" alt="LangChain" /></a>
  <a href="https://mistral.ai/"><img src="https://img.shields.io/badge/LLM-Mistral%20AI-FF7000" alt="Mistral AI" /></a>
  <a href="https://tavily.com/"><img src="https://img.shields.io/badge/Web%20Search-Tavily-635BFF" alt="Tavily" /></a>
</p>

**Live demo:** Coming soon

## Features

- **Live web search** to find relevant sources for a research topic.
- **Webpage content extraction** to collect readable information from selected URLs.
- **Four-stage research workflow** with separate responsibilities for search, reading, writing, and critique.
- **Structured report generation** with sections such as key findings, real-world applications, limitations, a conclusion, and sources where available.
- **AI-generated critique** with a score, strengths, and areas for improvement.
- **Research evidence and progress views** in an interactive Streamlit interface.
- **Report download** and a terminal-based pipeline execution option.

## Architecture and workflow

<p align="center">
  <img src="images/Research%20Agent%20Workflow.png" alt="ScrapeResearch.ai architecture showing the four workflow stages and their tools" width="95%" />
</p>

| Stage | Responsibility | Implementation |
|---|---|---|
| **1. Search Agent** | Finds relevant web sources for the given topic. | LangChain ReAct agent, `AgentExecutor`, custom `web_search` tool, Tavily API |
| **2. Reader Agent** | Visits selected URLs and extracts readable page content. | LangChain ReAct agent, `AgentExecutor`, custom `scrape_url` tool, Requests, Beautiful Soup, lxml |
| **3. Writer Chain** | Synthesizes the collected research into a structured report. | LangChain Expression Language (LCEL), prompt template, Mistral LLM, `StrOutputParser` |
| **4. Critic Chain** | Reviews the report and produces a score and written feedback. | LCEL, prompt template, Mistral LLM, `StrOutputParser` |

`pipeline.py` coordinates the stages and passes their intermediate results through a shared state dictionary. The Search and Reader stages are tool-using agents; the Writer and Critic stages are LCEL chains. Together, they form a multi-agent research pipeline without treating every stage as an autonomous agent.

## Technology stack

| Component | Technology |
|---|---|
| Language and interface | Python, Streamlit |
| Language model | Mistral AI, `langchain-mistralai` |
| Agents and orchestration | LangChain, ReAct, `AgentExecutor`, LCEL |
| Web search | Tavily API, `tavily-python` |
| Page extraction | Requests, Beautiful Soup, lxml |
| Output parsing | `StrOutputParser` |
| Configuration | `python-dotenv`, environment variables |

## Project structure

```text
ScrapeResearch-ai/
├── .streamlit/
│   └── config.toml
├── images/
│   ├── Critic Feedback.png
│   ├── End of Report Sources and Download Report.png
│   ├── Final Report Result.png
│   ├── Home Page.png
│   ├── More Report Results.png
│   ├── Research Agent Workflow.png
│   └── Research Completed Steps.png
├── .env.example
├── .gitignore
├── agents.py
├── app.py
├── pipeline.py
├── README.md
├── requirements.txt
└── tools.py
```

The local `.env` file contains API credentials and should not be committed to version control.

## Installation and setup

### 1. Clone the repository

```bash
git clone https://github.com/arjun11goel/multi-agent-research-pipeline.git
cd multi-agent-research-pipeline
```

### 2. Create a virtual environment and install dependencies

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

### 3. Configure API keys

Set the following variables in `.env`:

```dotenv
MISTRAL_API_KEY=your_mistral_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Mistral provides the language-model access, while Tavily provides web search. Both keys are required for the complete research workflow. Keep API credentials private.

## Run the application

**Streamlit web interface**

```bash
streamlit run app.py
```

**Terminal pipeline**

```bash
python pipeline.py
```

Enter a research topic when prompted by the terminal pipeline. The stages run in sequence, passing search results and extracted content to the writing and critique stages. Execution time and the amount of usable research depend on API responses and webpage availability.

## Screenshots

<details>
  <summary><strong>Research progress and completed workflow</strong></summary>
  <p><img src="images/Research%20Completed%20Steps.png" alt="Completed research workflow stages" width="100%" /></p>
</details>

<details>
  <summary><strong>Generated research report</strong></summary>
  <p><img src="images/Final%20Report%20Result.png" alt="Generated research report" width="100%" /></p>
  <p><img src="images/More%20Report%20Results.png" alt="Additional sections of the generated report" width="100%" /></p>
</details>

<details>
  <summary><strong>Critic feedback</strong></summary>
  <p><img src="images/Critic%20Feedback.png" alt="Critic score, strengths, and areas for improvement" width="100%" /></p>
</details>

<details>
  <summary><strong>Report sources and download</strong></summary>
  <p><img src="images/End%20of%20Report%20Sources%20and%20Download%20Report.png" alt="Report sources and download option" width="100%" /></p>
</details>

## Evaluation and limitations

The application includes an LLM-generated critique with a score and written feedback. This provides a qualitative review of the report, not an independently verified measure of factual accuracy. No quantitative benchmark results are currently reported.

- Generated claims may be incomplete or unsupported; important information should be checked against the cited sources.
- Webpage extraction depends on site availability and HTML structure. Some pages may block scraping or expose incomplete content.
- Search and generation depend on external APIs, network connectivity, and service limits.
- The Critic Chain can overlook factual errors, so its score should not be interpreted as a guarantee of report quality.

## Author

**Arjun Goel**

MCA'27@VIT Vellore

- **Email:** [arjun11goel@gmail.com](mailto:arjun11goel@gmail.com)
- **LinkedIn:** [linkedin.com/in/arjun11goel](https://www.linkedin.com/in/arjun11goel/)
- **Porfolio:** [arjun11goel.vercel.app](https://arjun11goel.vercel.app)

---

*Built as a Multi-Agent Generative AI project exploring live web research, webpage extraction, LLM-powered report generation, tool-using agents, LangChain LCEL orchestration, and automated report critique.*
