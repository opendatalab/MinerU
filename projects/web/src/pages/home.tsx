"use client";

import ErrorBoundary from "@/components/error-boundary";
import styles from "./home.module.scss";
import { SlotID, Path } from "@/constant/route";
import {
  BrowserRouter,
  Routes,
  Route,
  Outlet,
  Navigate,
  useLocation,
  HashRouter,
} from "react-router-dom";
import { ExtractorSide } from "./extract-side";
import { LanguageProvider } from "@/context/language-provider";
import PDFUpload from "@/pages/extract/components/pdf-upload";
import PDFExtractionJob from "@/pages/extract/components/pdf-extraction";

export function WindowContent() {
  const location = useLocation();
  const isHome = location.pathname === Path.Home;

  return (
    <>
      <ExtractorSide className={isHome ? styles["sidebar-show"] : ""} />
      <div className="flex-1">
        <Outlet />
      </div>
    </>
  );
}

function Screen() {
  const renderContent = () => {
    return (
      <div className="w-full h-full flex" id={SlotID.AppBody}>
        <Routes>
          <Route path="/" element={<WindowContent />}>
            <Route
              index
              element={<Navigate to="/OpenSourceTools/Extractor/PDF" replace />}
            />
            <Route
              path="/OpenSourceTools/Extractor/PDF"
              element={<PDFUpload />}
            />
            <Route
              path="/OpenSourceTools/Extractor/PDF/:jobID"
              element={<PDFExtractionJob />}
            />
            <Route
              path="*"
              element={<Navigate to="/OpenSourceTools/Extractor/PDF" replace />}
            />
          </Route>
        </Routes>
      </div>
    );
  };

  return <>{renderContent()}</>;
}

export function Home() {
  return (
    <ErrorBoundary>
      <LanguageProvider>
        <HashRouter>
          <Screen />
        </HashRouter>
      </LanguageProvider>
    </ErrorBoundary>
  );
}
