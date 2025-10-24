@echo off
set /p ALIAS="캡셔닝 할 캐릭터 또는 풍경의 고유 이름을 입력하세요: "
python gen-cap.py --dirs ../dataset/train/mainchar/2_karina --char "%ALIAS%" --device "cuda:3" --overwrite

pause