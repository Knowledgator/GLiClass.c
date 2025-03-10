# Check if the first argument (model name) is provided
if ($args.Length -lt 1) {
    Write-Host "You need to specify a model name e.g: .\run_GLiClass.ps1 knowledgator/gliclass-base-v1.0"
    exit 1
}

# Check if the second argument (path to .json file) is provided
if ($args.Length -lt 2) {
    Write-Host "You need to specify a path to the .json file e.g: .\run_GLiClass.ps1 knowledgator/gliclass-base-v1.0 C:\path\to\your_data.json"
    exit 1
}

# Assign arguments to variables
$MODEL_NAME = $args[0]
$JSON_FILE_PATH = $args[1]

# Dirs
$TOKENIZER_DIR = "tokenizer"
$MODEL_DIR = "onnx"

# Create directories
New-Item -ItemType Directory -Force -Path $MODEL_DIR
New-Item -ItemType Directory -Force -Path $TOKENIZER_DIR

# Files
$TOKENIZER_FILE = "$TOKENIZER_DIR\tokenizer.json"
$MODEL_CONFIG_FILE = "$MODEL_DIR\config.json"
$MODEL_ONNX_FILE = "$MODEL_DIR\model.onnx"

# URLs
$MODEL_CONFIG_URL = "https://huggingface.co/$MODEL_NAME/resolve/main/onnx/config.json"
$MODEL_ONNX_URL = "https://huggingface.co/$MODEL_NAME/resolve/main/onnx/model.onnx"
$TOKENIZER_URL = "https://huggingface.co/$MODEL_NAME/raw/main/tokenizer.json"

# Function to download a file
function Download-File {
    param(
        [string]$FileType,
        [string]$Directory,
        [string]$Url
    )

    Write-Host "Downloading $FileType..."
    Invoke-WebRequest -Uri $Url -OutFile "$Directory\$(Split-Path -Leaf $Url)"
}

# Function to download model files
function Download-Model {
    Remove-Item -Force $TOKENIZER_FILE
    Remove-Item -Force $MODEL_CONFIG_FILE
    Remove-Item -Force $MODEL_ONNX_FILE

    Start-BitsTransfer -Destination $MODEL_DIR -Source $MODEL_CONFIG_URL
    Start-BitsTransfer -Destination $MODEL_DIR -Source $MODEL_ONNX_URL
    Start-BitsTransfer -Destination $TOKENIZER_DIR -Source $TOKENIZER_URL
}

# Download logic
if (-not (Test-Path $MODEL_CONFIG_FILE)) {
    Write-Host "File $MODEL_CONFIG_FILE was not found."
    Download-Model
} else {
    $MODEL_TYPE = (Get-Content $MODEL_CONFIG_FILE | ConvertFrom-Json).original_model_name
    if ($MODEL_TYPE -ne $MODEL_NAME) {
        Write-Host "Reconfiguring the model"
        Download-Model
    } else {
        Write-Host "Checking the integrity of model files"
        $FILES = @($TOKENIZER_FILE, $MODEL_ONNX_FILE)
        foreach ($FILE in $FILES) {
            if (-not (Test-Path $FILE)) {
                Write-Host "Missing file: $FILE. Downloading..."
                if ($FILE -eq $TOKENIZER_FILE) {
                    Start-BitsTransfer -Source $TOKENIZER_URL -Destination $TOKENIZER_DIR
                } elseif ($FILE -eq $MODEL_ONNX_FILE) {
                    Start-BitsTransfer  -Destination $MODEL_DIR -Source $MODEL_ONNX_URL
                }
            }
        }
    }
    Write-Host "Everything was set up. Running inference"
}

# Validate prompt_first field in model config
$PROMPT_FIRST = (Get-Content $MODEL_CONFIG_FILE | ConvertFrom-Json).prompt_first
Write-Output $PROMPT_FIRST
if ($PROMPT_FIRST -ne $true -and $PROMPT_FIRST -ne $false) {
    Write-Host "Something wrong with model configuration file."
    Write-Host "Expected values: 'true' or 'false' but received: $PROMPT_FIRST"
    exit 1
}

# Run the inference command
Start-Process -FilePath ".\build\Release\GLiClass.exe" -ArgumentList $JSON_FILE_PATH, $PROMPT_FIRST