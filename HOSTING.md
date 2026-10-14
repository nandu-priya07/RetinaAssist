# Hosting RetinaAssist with Render and Vercel

## Architecture

RetinaAssist is a single Flask application. Flask renders the pages and handles sign-in, uploads, model inference, records, and reports. Keep that application on Render. Vercel will proxy the same site under your Vercel domain; it is not a separately built frontend. A true split frontend/backend needs a frontend rewrite and API/authentication work.

## 1. Push the project to GitHub

Commit and push this repository, including models/retinaassist_efficientnet_finetuned.keras. Do not commit .env or any real patient images. The Docker build checks the model checksum and needs the model file in the repository.

## 2. Create the Render service

1. In Render, choose New > Blueprint, connect the GitHub repository, and apply the render.yaml Blueprint.
2. The Blueprint builds the repository Dockerfile, sets production mode and a persistent disk at /var/data, and generates the session signing key.
3. Enter values for RETINAASSIST_DEMO_PASSWORD and RETINAASSIST_ADMIN_PASSWORD when prompted. Use strong, different passwords. The demo password is shared by the doctor and staff demo accounts; keep the admin password private.
4. The service uses Render Standard (2 GB RAM) because TensorFlow needs more memory than the small 512 MB plans. The persistent disk is required because SQLite, uploaded images, and reports are stored on disk. This setup requires a paid Render service/disk.
5. Wait for the first build and deploy. It downloads Python dependencies, verifies the model, then loads the model at startup; the first deploy can take several minutes.
6. Open the Render service URL, such as https://retinaassist.onrender.com. Check that the home page loads and sign-in works. Use synthetic patient codes and authorized test images only.

Render may ask to create or confirm the Blueprint's sync:false secrets in the dashboard. If so, add the two password values in Environment and redeploy. Keep RETINAASSIST_ENV=production, RETINAASSIST_DATA_DIR=/var/data, and the generated RETINAASSIST_SECRET_KEY.

## 3. Point Vercel at Render

1. Open vercel.json in this repository and replace YOUR-RENDER-SERVICE with your actual Render service name. For example, if the Render URL is https://retinaassist.onrender.com, set the destination to https://retinaassist.onrender.com/:path*.
2. Commit and push that change.
3. In Vercel, choose Add New > Project, import the same GitHub repository, and deploy. This is a proxy deployment: requests for pages, forms, static assets, and downloads are forwarded to Render.
4. In Vercel project settings, set the Framework Preset to Other and leave the build command and output directory empty. The Flask application does not run on Vercel.
5. After deployment, open the Vercel URL and test navigation, login/logout, image upload, and report download. The address should remain on Vercel while the app runs on Render.

## 4. Optional custom domain

Add your domain to the Vercel project and follow Vercel's DNS instructions. The Vercel domain becomes the public URL; leave the Render URL available for troubleshooting.

## Updating the app

Push commits to GitHub. Render rebuilds/redeploys the backend, and Vercel redeploys the proxy configuration. Data under /var/data survives restarts and deploys on the attached disk. Keep a separate backup of that disk/database and uploaded files.

## Important limitations

- The app is a research prototype and is not clinically validated or approved for diagnostic use. Do not upload real patient data.
- Render disks are attached to one service instance; this SQLite-based setup is a single-instance deployment.
- If you later change the Render service name, update the Vercel rewrite destination too.
