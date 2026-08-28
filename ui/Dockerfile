# Admin console image: static Vite build served by nginx.
#
# The backend URL is a RUNTIME setting, not a build arg: the bundle makes
# same-origin /admin/... calls (VITE_API_BASE_URL stays empty) and nginx
# proxies /admin/ to $BACKEND_URL, so one image serves every environment.
# See example.env for the variables.

FROM node:22-alpine AS build
WORKDIR /app
RUN npm install -g pnpm@10
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY . .
# Build-time knobs (baked into the bundle). Leave VITE_API_BASE_URL empty so
# the nginx proxy below handles routing; set VITE_LANGFUSE_URL to show
# "open in Langfuse" links.
ARG VITE_API_BASE_URL=""
ARG VITE_LANGFUSE_URL=""
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL \
    VITE_LANGFUSE_URL=$VITE_LANGFUSE_URL
RUN pnpm build

FROM nginx:1.29-alpine
COPY --from=build /app/dist /usr/share/nginx/html
# The stock nginx entrypoint envsubsts /etc/nginx/templates/*.template into
# /etc/nginx/conf.d/ at container start, substituting only variables that are
# set in the environment — nginx's own $uri/$host survive untouched.
COPY nginx/default.conf.template /etc/nginx/templates/default.conf.template
# Where /admin/*, /agents and /healthz are proxied — the one setting a
# deployment has to get right. Scheme + host + port only: no trailing slash, no
# path. In compose this is the backend *service name*, not localhost (localhost
# inside this container is nginx itself):
#
#     services:
#       ui:
#         image: sciedu-llm-ui
#         environment:
#           BACKEND_URL: http://llm-provider:8080
#
# The name is resolved per request, so the backend may start after this
# container and may be recreated on a new IP — see nginx/default.conf.template.
ENV BACKEND_URL=http://localhost:8080
# The DNS server that resolution goes to. 127.0.0.11 is Docker's embedded DNS,
# which is what serves compose service names; override it when running anywhere
# that is not Docker.
ENV DNS_RESOLVER=127.0.0.11
EXPOSE 80
