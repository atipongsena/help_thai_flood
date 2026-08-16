param (
    [Parameter(Mandatory=$true)]
    [string]$MongoUri,
    [string]$ProjectId = "my-flood-project-480015",
    [string]$Region = "asia-southeast1"
)

$ErrorActionPreference = "Stop"

Write-Host "Starting Deployment..." -ForegroundColor Green

# 1. Deploy API
Write-Host "`n[1/3] Deploying API Service..." -ForegroundColor Cyan
Set-Location "d:\help_thai_flood"
gcloud builds submit --config cloudbuild_api.yaml .
gcloud run deploy flood-api --image "gcr.io/$ProjectId/flood-api" --platform managed --region $Region --allow-unauthenticated --memory 8Gi --timeout 600s

# Get API URL
$ApiUrl = gcloud run services describe flood-api --platform managed --region $Region --format 'value(status.url)'
Write-Host "API Deployed at: $ApiUrl" -ForegroundColor Green

# 2. Deploy Server
Write-Host "`n[2/3] Deploying Backend Server..." -ForegroundColor Cyan
Set-Location "d:\help_thai_flood\server"
gcloud builds submit --config cloudbuild_server.yaml .
gcloud run deploy flood-server --image "gcr.io/$ProjectId/flood-server" --platform managed --region $Region --allow-unauthenticated --set-env-vars "MONGO_URI=$MongoUri,MODEL_API_URL=$ApiUrl/api/predict"

# Get Server URL
$ServerUrl = gcloud run services describe flood-server --platform managed --region $Region --format 'value(status.url)'
Write-Host "Server Deployed at: $ServerUrl" -ForegroundColor Green

# 3. Deploy Web
Write-Host "`n[3/3] Deploying Web App..." -ForegroundColor Cyan
Set-Location "d:\help_thai_flood\web"

# Update vercel.json
$VercelConfig = @{
    rewrites = @(
        @{
            source = "/api/:path*"
            destination = "$ServerUrl/api/:path*"
        }
    )
}
$VercelConfig | ConvertTo-Json -Depth 4 | Out-File "vercel.json" -Encoding utf8

# Deploy
vercel --prod

Write-Host "`nDeployment Complete!" -ForegroundColor Green
Write-Host "API: $ApiUrl"
Write-Host "Server: $ServerUrl"
