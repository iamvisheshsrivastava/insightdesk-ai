FROM node:18-alpine AS build
WORKDIR /app
COPY package*.json ./
COPY vite.config.js ./
COPY src ./src
COPY index.html ./
RUN npm ci --silent && npm run build

FROM nginx:stable-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
