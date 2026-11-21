@echo off
rem Navapp-chatbot - dubbelklik om te starten (gebaseerd op Start_VaidyaBot.bat).
rem Sluit het app-venster om te stoppen; de server stopt dan mee.
setlocal
cd /d "%~dp0"
set PORT=8766
set URL=http://localhost:%PORT%

rem eigen python\ (release) of .venv\, anders tijdelijk de omgeving van VaidyaBot
set PY=%~dp0python\python.exe
if not exist "%PY%" set PY=%~dp0.venv\Scripts\python.exe
if not exist "%PY%" set PY=%USERPROFILE%\Sanskriet\.venv\Scripts\python.exe

curl -s -o NUL --max-time 1 %URL% >NUL 2>&1 && goto :open

echo Navapp-chatbot wordt gestart, een ogenblik geduld...
start "Navapp-chatbot-server" /min "%PY%" -m chainlit run app\chainlit_app.py --port %PORT% -h

:wait
timeout /t 1 /nobreak >NUL
curl -s -o NUL --max-time 1 %URL% >NUL 2>&1 || goto :wait

:open
set EDGE=%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe
if not exist "%EDGE%" set EDGE=%ProgramFiles%\Microsoft\Edge\Application\msedge.exe
if exist "%EDGE%" (
  start "" /wait "%EDGE%" --app=%URL% --user-data-dir="%LOCALAPPDATA%\NavappBot"
  powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }" >NUL 2>&1
) else (
  start "" %URL%
  echo Sluit dit venster om de chatbot te stoppen.
  pause >NUL
)
endlocal
