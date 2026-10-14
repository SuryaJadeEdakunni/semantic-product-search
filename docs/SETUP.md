# Setup & Troubleshooting

Detailed setup for Windows and macOS. For the short version, see the [Quick start](../README.md#quick-start) in the main README.

**Before you start:** clone the repository (or download it as a ZIP and extract it), then download `chroma_db.zip` from the [Releases](../../releases) page and unzip it so that the project looks like this:

```text
semantic-product-search/
├── main.py
├── requirements.txt
├── recommender/
├── scripts/
└── chroma_db/
    ├── chroma.sqlite3
    └── [ChromaDB index folder(s)]
```

Do not rename, move, or edit files inside `chroma_db`.

## What the final product does

The program follows this pipeline:

1. The user types a natural-language beauty-product request.
2. `recommender/understand_query.py` uses Llama 3.1 through Ollama to extract the descriptive search meaning.
3. `recommender/search.py` embeds that semantic query using `all-MiniLM-L6-v2`.
4. ChromaDB searches the `SephoraProducts` collection and retrieves 10 semantically similar candidates.
5. `recommender/response.py` sends the retrieved candidates to Llama 3.1 through Ollama.
6. The program displays 1–3 readable recommendations.

The stored product embeddings are already inside the ChromaDB database. The program only embeds each **new user query** when it runs.

---

## Software required

You need:

- Python 3
- Internet access during initial setup so Python packages and the Sentence Transformer model can be downloaded
- Ollama installed locally
- The Ollama model `llama3.1`
- A terminal / command prompt

VS Code is optional. The program can be run entirely from Terminal, PowerShell, or Command Prompt.

### Older computers / older Macs

The project runs local AI software, so an older machine can create setup or performance issues.

If Ollama cannot be installed or started on the machine, the final application cannot run as written because both the query-interpretation and final-response stages call a local Ollama model.

If package installation fails on a very new or very old Python version, try Python 3.11 or 3.12 in a fresh virtual environment.

---

## Open a terminal in the correct folder

The terminal must be inside `semantic-product-search`.

### Windows

Option A:
1. Open the `semantic-product-search` folder in File Explorer.
2. Click the address bar.
3. Type `powershell`
4. Press Enter.

Option B:
```powershell
cd "C:\path\to\semantic-product-search"
```

Check:
```powershell
dir
```

You should see:
- `main.py`
- `recommender/understand_query.py`
- `recommender/search.py`
- `recommender/response.py`
- `chroma_db`

### macOS

Open Terminal and move into the downloaded folder:

```bash
cd "/path/to/semantic-product-search"
```

For example:

```bash
cd ~/Documents/semantic-product-search
```

Check:

```bash
ls
```

You should see:
- `main.py`
- `recommender/understand_query.py`
- `recommender/search.py`
- `recommender/response.py`
- `chroma_db`

If you do not see those files, stop and move to the correct folder before continuing.

---

## Create a virtual environment

A virtual environment prevents package conflicts with other Python projects.

### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If `python` is not recognized, try:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

After activation, the terminal will usually show `(.venv)` at the beginning of the line.

---

## Install the Python packages

With the virtual environment activated:

### Windows

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

The important runtime packages are:
- `chromadb`
- `sentence-transformers`
- `ollama`
- `pydantic`

The first installation can take time because `sentence-transformers` installs machine-learning dependencies.

---

## Install and prepare Ollama

Ollama is a separate application. Installing the Python package `ollama` is **not enough**.

1. Install the Ollama desktop/application version for your operating system.
2. Open/start Ollama.
3. Open a terminal.
4. Pull the required model:

```bash
ollama pull llama3.1
```

5. Test that the model is available:

```bash
ollama list
```

You should see a Llama 3.1 model.

Optional direct test:

```bash
ollama run llama3.1
```

Type a short message. If it responds, Ollama is working.

Exit that test before running the project.

### If `ollama: command not found`

Ollama is either:
- not installed,
- not available on the system PATH,
- or the terminal needs to be restarted after installation.

### If Ollama installs but the model will not run

The machine may not have enough resources, or the operating system/hardware may not support the installed Ollama version well. Try the project on a newer machine.

---

## Confirm the ChromaDB folder exists

Before starting the application, verify this path exists:

```text
semantic-product-search/chroma_db/
```

It must contain:

```text
chroma.sqlite3
```

and ChromaDB's generated vector-index folder(s)/files.

Do not replace `chroma_db` with an empty folder. If it is missing, download `chroma_db.zip` from the Releases page and unzip it into the project root.

If you downloaded `chroma_db.zip` from Releases you do not need to run anything in `scripts/`. Only use `scripts/build_dataset.py` and `scripts/build_index.py` if you want to rebuild the database from the raw data.

---

## Run the application

Make sure:
- you are inside the `semantic-product-search` folder;
- the virtual environment is activated;
- the packages are installed;
- Ollama is running;
- `llama3.1` has been pulled;
- `chroma_db` is present.

Then run:

### Windows

```powershell
python main.py
```

If needed:

```powershell
py main.py
```

### macOS

```bash
python3 main.py
```

The terminal should display:

```text
Enter your query (or type 'exit' to quit):
```

Example:

```text
I'm looking for a warm, cozy perfume for winter evenings.
```

The program should return 1–3 product recommendations.

To stop the application:

```text
exit
```

---

## The first run may be slower

`recommender/search.py` loads the Sentence Transformer model:

```text
all-MiniLM-L6-v2
```

On a machine that has never used this model before, the model files may need to be downloaded and cached first.

This means:
- the first run may take longer;
- internet access may be required for the first model download;
- later runs should use the locally cached model.

Ollama inference can also be slow on older hardware.

---

## Common errors

### Error: `No module named ...`

Example:
```text
ModuleNotFoundError: No module named 'chromadb'
```

Cause: packages were not installed in the Python environment currently being used.

Fix:
1. Activate `.venv`.
2. Run:
```bash
python -m pip install -r requirements.txt
```

On macOS, if required:
```bash
python3 -m pip install -r requirements.txt
```

---

### Error: cannot connect to Ollama / connection refused

Cause: the Ollama application/server is not running.

Fix:
1. Open Ollama.
2. Test:
```bash
ollama list
```
3. Run the project again.

---

### Error: model `llama3.1` not found

Fix:

```bash
ollama pull llama3.1
```

Then:

```bash
ollama list
```

---

### Error: ChromaDB collection does not exist

Example:
```text
Collection SephoraProducts does not exist
```

Likely causes:
- `chroma_db` is missing,
- only some project files were downloaded,
- the program is being run from the wrong working directory,
- the supplied database folder structure was changed.

Fix:
1. Confirm the terminal is inside `semantic-product-search`.
2. Confirm `semantic-product-search/chroma_db/chroma.sqlite3` exists.
3. Re-download/extract the complete `chroma_db.zip` from the Releases page if necessary.

Alternatively, rebuild it with `python scripts/build_dataset.py` followed by `python scripts/build_index.py`.

---

### Error mentioning `chroma.sqlite3`, segment files, or index files

The ChromaDB database is incomplete or its internal file structure has been changed.

Fix:
- Replace the entire `chroma_db` folder with a fresh copy from the GitHub Release (`chroma_db.zip`).
- Do not copy only `chroma.sqlite3`; the associated ChromaDB index folder(s) are also part of the database.

---

### Error: `python` command not found

### Windows
Try:
```powershell
py --version
```

### macOS
Try:
```bash
python3 --version
```

If neither works, install Python.

---

### Error installing packages on an older Mac

Possible causes include:
- outdated macOS,
- unsupported hardware,
- incompatible Python version,
- machine-learning dependencies without a compatible build.

Try:
1. Install Python 3.11 or 3.12.
2. Delete the old `.venv`.
3. Create a new `.venv`.
4. Re-run `pip install -r requirements.txt`.

If Ollama itself does not support the computer, use another computer because Ollama is required by the final pipeline.

---

### Program opens but is extremely slow

The program uses:
- a Sentence Transformer locally;
- ChromaDB locally;
- Llama 3.1 through Ollama locally.

Performance therefore depends heavily on the computer.

The first run can also be slower because models may be loading or downloading.

---

## Quick-start version

Once setup has been completed successfully, future runs are much shorter.

### Windows

```powershell
cd "C:\path\to\semantic-product-search"
.\.venv\Scripts\Activate.ps1
python main.py
```

### macOS

```bash
cd "/path/to/semantic-product-search"
source .venv/bin/activate
python3 main.py
```

Ollama must still be running.

---

## Troubleshooting checklist

Before assuming there is a code problem, verify all seven:

- [ ] The complete project folder was downloaded locally.
- [ ] Terminal is currently inside `semantic-product-search`.
- [ ] `.venv` is activated.
- [ ] `requirements.txt` installed successfully.
- [ ] Ollama is installed and running.
- [ ] `llama3.1` appears in `ollama list`.
- [ ] `chroma_db/chroma.sqlite3` and the ChromaDB index files are present.

If all seven are true, run:

```bash
python main.py
```

or on macOS:

```bash
python3 main.py
```
