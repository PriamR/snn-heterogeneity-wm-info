@echo off
cd /d "C:\Users\Priya\Desktop\research project (SNN Info Theory)\Project Files"
"C:\Users\Priya\AppData\Local\Programs\Python\Python311\python.exe" "Seeded 4 Class\Seeded run 1 4class.py" > "Seeded 4 Class\training_run1_log.txt" 2> "Seeded 4 Class\training_run1_errors.txt"
echo Exit code: %ERRORLEVEL% >> "Seeded 4 Class\training_run1_log.txt"
