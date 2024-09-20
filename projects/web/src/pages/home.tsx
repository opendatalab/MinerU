"use client";

import ErrorBoundary from "@/components/error-boundary";
import styles from "./home.module.scss";
import { SlotID, Path } from "@/constant/route";
import { HashRouter, Routes, Route, Outlet } from "react-router-dom";
import { ExtractorSide } from "./extract-side";
import { LanguageProvider } from "@/context/language-provider";
import PDFUpload from "@/pages/extract/components/pdf-upload";
import PDFExtractionJob from "@/pages/extract/components/pdf-extraction";

// judge if the app has hydrated
// const useHasHydrated = () => {
//   const [hasHydrated, setHasHydrated] = useState<boolean>(false);
//   useEffect(() => {
//     setHasHydrated(true);
//   }, []);
//   return hasHydrated;
// };

export function WindowContent() {
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
  // if you do not need to use the renderContent for rendering router, you can use the other render function to interrupt before the renderContent

  const renderContent = () => {
    return (
      <div className="w-full h-full flex" id={SlotID.AppBody}>
        <Routes>
          <Route path="/" element={<WindowContent />}>
            <Route
              path="/OpenSourceTools/Extractor/PDF"
              element={<PDFUpload />}
            />
            <Route
              path="/OpenSourceTools/Extractor/PDF/:jobID"
              element={<PDFExtractionJob />}
            />
            {/* <Route path="*" element={<PDFUpload />} /> */}
          </Route>
        </Routes>
      </div>

      // <ExtractorSide className={isHome ? styles["sidebar-show"] : ""} />
      // <WindowContent className="flex-1">
      //   <AppRoutes />
      // </WindowContent>
    );
  };

  return <>{renderContent()}</>;
}

export function Home() {
  // leave this comment to check if the app has hydrated
  // if (!useHasHydrated()) {
  //   return <LoadingAnimation />;
  // }

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
