# Deploying Numbl Execution Service on DigitalOcean

## 1. Create the Droplet

- Ubuntu, 4GB RAM or more recommended
- SSH in as root

## 2. Install Dependencies

```bash
# Update system
apt update && apt upgrade -y

# Install Node.js 22
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt install -y nodejs

# Install build tools (for native addons)
apt install -y build-essential python3

# Install Git and GitHub CLI
apt install -y git
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list
apt update && apt install -y gh

# Install OpenBLAS and FFTW3 (for the native LAPACK/FFT addon)
apt install -y libopenblas-dev libfftw3-dev
```

## 3. Clone and Build

```bash
# Authenticate with GitHub
gh auth login

# Clone the repo
gh repo clone flatironinstitute/numbl
cd numbl

# Install npm dependencies
npm install

# Build everything
npm run build:addon   # native LAPACK addon (requires libopenblas-dev)
npm run build:cli     # numbl CLI
npm run build:server  # execution service

# Configure the service
cp server/config.example.env server/.env
# Edit server/.env if you want to change defaults (port, timeout, memory, concurrency)
```

## 4. Test the Service

```bash
# Start it in dev mode to verify
npm run server:dev
# Should print: Numbl execution service running on port 3001

# In another terminal, test it
curl http://localhost:3001/health
# Expected: {"status":"ok","activeExecutions":0,"maxConcurrentExecutions":3}

# Test a script execution
curl -X POST http://localhost:3001/execute \
  -H "Content-Type: application/json" \
  -d '{"files":[{"name":"test.m","content":"disp(\"hello\");"}],"mainScript":"test.m"}'

# Stop the dev server (Ctrl+C) when done testing
```

## 5. Set Up PM2

```bash
# Install PM2
npm install -g pm2

# Edit the cwd in ecosystem.config.cjs to match your project path
nano server/ecosystem.config.cjs

# Start using the ecosystem config (loads server/.env automatically)
pm2 start server/ecosystem.config.cjs

# Persist across reboots
pm2 save
pm2 startup
# Run the command it prints (starts with "sudo env PATH=...")
```

## 6. Configure Firewall

```bash
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable
ufw status
```

## 7. Set Up Nginx

```bash
apt install -y nginx
```

Create `/etc/nginx/sites-available/numbl`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:3001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 60s;
        proxy_connect_timeout 60s;

        # Disable proxy buffering for SSE streaming
        proxy_buffering off;
    }
}
```

```bash
# Enable the site
ln -s /etc/nginx/sites-available/numbl /etc/nginx/sites-enabled/
nginx -t
systemctl restart nginx
```

## 8. Set Up HTTPS with Let's Encrypt

**First, in Cloudflare (or your DNS provider):**

- Add an A record pointing your domain/subdomain to the droplet's IP
- Set proxy status to **DNS only (grey cloud)** — required for Let's Encrypt

**Verify DNS has propagated:**

```bash
dig your-domain.com +short
# Should return your droplet's IP
```

**Get the certificate:**

```bash
apt install -y certbot python3-certbot-nginx
certbot --nginx -d your-domain.com
```

Certbot automatically modifies the Nginx config to handle HTTPS and sets up auto-renewal.

**Test HTTPS:**

```bash
curl https://your-domain.com/health
```

## 9. Connect the Web UI

In the browser console on the Numbl web UI:

```javascript
localStorage.setItem("numbl_remote_service_url", "https://your-domain.com");
```

Then use the **Local / Remote** toggle in the IDE to switch to remote execution.

## Useful Commands

```bash
# View service logs
pm2 logs numbl-service

# Restart service
pm2 restart numbl-service

# Check status
pm2 status
```

## Updating After Code Changes

```bash
cd ~/numbl
git pull
npm run build:addon   # only if native code changed
npm run build:cli
npm run build:server
pm2 restart numbl-service  # picks up server/.env automatically via ecosystem config
```
