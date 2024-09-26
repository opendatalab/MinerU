# MinerU web

## Table of Contents
- [Local Frontend Development](#local-frontend-development)
- [Technology Stack](#technology-stack)

## Local Frontend Development

### Prerequisites
- Node.js 18.x
- pnpm

### Installation Steps

1. Install Node.js 18
   - Visit the [Node.js official website](https://nodejs.org/) to download and install Node.js version 18.x

2. Install pnpm
   ```bash
   npm install -g pnpm
3. Clone the repository
    ```git clone https://github.com/opendatalab/MinerU
    cd ./projects/web
    ```
4. Install dependencies
    ```
    pnpm install
    ```
5. Run the development server
    ```
    pnpm run dev
    ```
6. ⚠️ Note: This command is for local development only, do not use for deployment!
Open your browser and visit http://localhost:5173 (or another address output in the console)

7. Ensure that the backend service in ./projects/web_demo is running

8. If you encounter an error when executing `pnpm install`, you can switch to an alternative package manager.
   ```
   npm install -g yarn
   yarn
   yarn start
   ```


##  Building the Project
```
pnpm run build
```
## Technology Stack
- React
- Tailwind CSS
- typeScript
- zustand
- ahooks
