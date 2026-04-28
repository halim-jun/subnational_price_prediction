# Cloudflare Pages Prototype Deployment

> **이 가이드는 프로토타입 전용입니다.** 본 빌드는 데모/미리보기 목적이며 프로덕션이 아닙니다. 강한 보안을 보장하지 않으며, 단일 공유 비밀번호로만 접근을 차단합니다.

## 무엇이 다른가

`src/dashboard/frontend/` 의 일반적인 로컬 개발(`npm run dev` + FastAPI)과 달리, **Cloudflare 프로토타입 빌드**는:

- **백엔드 없음** — FastAPI 대신 정적 JSON 파일에서 데이터 읽음 (`public/data/*.json`)
- **로그인 장벽** — Cloudflare Pages Functions가 모든 요청을 가로채서 비밀번호 검증
- **자동 갱신 안 됨** — 새 모델 결과를 반영하려면 export 스크립트 재실행 후 재배포

## 빌드 절차

```bash
# 1. 데이터를 정적 JSON으로 export
python scripts/export_static_data.py
#   → src/dashboard/frontend/public/data/ 에 ~40개 파일 생성

# 2. Static export 빌드 (NEXT_PUBLIC_STATIC_MODE=true 필수)
cd src/dashboard/frontend
NEXT_PUBLIC_STATIC_MODE=true npm run build
#   → out/ 디렉토리 생성

# 3. Cloudflare Pages에 배포
#    빌드 출력 디렉토리: src/dashboard/frontend/out
#    Functions 디렉토리: src/dashboard/frontend/functions (자동 인식)
```

## Cloudflare Pages 환경변수

Cloudflare Pages 프로젝트 설정 > Environment variables에 추가:

| 이름 | 설명 | 예시 |
|---|---|---|
| `SITE_PASSWORD` | 공유 로그인 비밀번호 | `my-prototype-2026` |
| `AUTH_SECRET` | 쿠키 HMAC 서명용 32+ 글자 랜덤 문자열 | `openssl rand -hex 32` 결과 |

**둘 다 Production / Preview 양쪽 모두에 설정해야 합니다.**

`AUTH_SECRET` 생성 예:
```bash
openssl rand -hex 32
# 또는
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

## 인증 동작

1. 사용자가 `https://<your-domain>/` 접속
2. `functions/_middleware.ts`가 쿠키 확인 → 없으면 `/login`으로 302 redirect
3. 사용자가 비밀번호 제출 → `functions/api/auth/login.ts`가 검증 → HMAC 서명 쿠키 발급
4. 쿠키 7일 유효, `HttpOnly; Secure; SameSite=Lax`
5. 5회 연속 실패 시 60초 IP 락아웃 (in-memory, per-Worker-isolate)
6. 사이드바 하단 "로그아웃" 버튼으로 즉시 쿠키 삭제

## 로컬 개발 영향 없음

```bash
# 로컬 개발은 변경 없음 — STATIC_MODE 미설정이면 기존 FastAPI 사용
uvicorn src.dashboard.api.main:app --reload --port 8000
cd src/dashboard/frontend && npm run dev
```

## 향후 프로덕션화

이 빌드는 임시 차단막입니다. 프로덕션 전환 시 권장:

1. FastAPI 백엔드를 Railway/Fly.io/Render 등에 배포
2. `NEXT_PUBLIC_STATIC_MODE` 제거 (또는 `false`로 설정)
3. 공유 비밀번호 → **Cloudflare Access** 또는 Auth0 등 정식 IdP로 교체
4. `functions/api/auth/*` 와 `_middleware.ts` 제거 (IdP가 대체)
5. `public/data/` 정리

## 파일 위치 요약

```
docs/cloudflare-prototype.md                              ← 이 문서
docs/superpowers/specs/2026-04-28-cloudflare-prototype-login-design.md  ← 설계 spec
scripts/export_static_data.py                             ← 데이터 export
src/dashboard/frontend/
├── functions/
│   ├── _middleware.ts                                    ← 인증 게이트
│   ├── _lib/auth.ts                                      ← HMAC 헬퍼
│   └── api/auth/{login,logout}.ts
├── src/app/login/page.tsx                                ← 로그인 페이지
├── src/components/{AppShell,LogoutButton}.tsx
├── src/lib/api.ts                                        ← STATIC_MODE 분기
├── public/data/                                          ← export된 JSON (gitignored)
└── .env.local.example                                    ← 환경변수 템플릿
```
