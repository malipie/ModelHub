FROM nginx:1.27-alpine

# Remove default config
RUN rm /etc/nginx/conf.d/default.conf

COPY gateway/nginx.conf /etc/nginx/conf.d/api_gateway.conf

# Note: nginx -t would fail at build time because upstream hostnames
# (serving-champion, serving-challenger) only exist in the Docker Compose network.
# Config is validated at container startup instead.

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
