import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ProjectListPage } from "./pages/ProjectListPage";
import { ProjectIDEPage } from "./pages/ProjectIDEPage";
import { ShareIDEPage } from "./pages/ShareIDEPage";
import { EmbedPage } from "./pages/EmbedPage";
import { EmbedReplPage } from "./pages/EmbedReplPage";
import { BackendBenchPage } from "./pages/BackendBenchPage";

function App() {
  console.log("App initialized");
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ProjectListPage />} />
        <Route path="/project/:projectName" element={<ProjectIDEPage />} />
        <Route path="/share" element={<ShareIDEPage />} />
        <Route path="/embed" element={<EmbedPage />} />
        <Route path="/embed-repl" element={<EmbedReplPage />} />
        <Route path="/bench" element={<BackendBenchPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
