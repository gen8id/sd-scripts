@echo off
set /p ALIAS="캡셔닝 할 캐릭터 또는 풍경의 고유 이름을 영문과 숫자 조합으로 입력하세요. 특수문자는 안되며, 'robot, alice, human, tom' 등 일반적으로 사용되는 키워드는 학습에 실패하기 때문에 유니크한 문자, 또는 문자 뒤에 숫자를 붙여 고유한 문자열로 입력 바랍니다: "
python gen-cap.py --char "%ALIAS%" --device "cuda:3" --overwrite

pause