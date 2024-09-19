import { Routes, Route } from "react-router-dom";
import PDFUpload from "@/pages/extract/components/pdf-upload";
import PDFExtractionJob from "@/pages/extract/components/pdf-extraction";

function AppRoutes() {
  return (
    <>
      <Route path="/OpenSourceTools/Extractor/PDF" element={<PDFUpload />} />
      <Route
        path="/OpenSourceTools/Extractor/PDF/:jobID"
        element={<PDFExtractionJob />}
      />
    </>
  );
}

export default AppRoutes;
