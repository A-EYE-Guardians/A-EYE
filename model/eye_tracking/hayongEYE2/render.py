import inspect
from l2cs import render

# 1) 소스 파일 경로 확인
print(render.__module__)            # 모듈 이름
print(inspect.getsourcefile(render))# 소스 파일 경로(가능하면)
print(inspect.getfile(render))      # 파일 경로(바이트코드라도 경로는 대개 나옴)

# 2) 소스 내용 출력
print(inspect.getsource(render))    # <-- 여기서 소스가 그대로 출력됨 (순수 파이썬일 때)
