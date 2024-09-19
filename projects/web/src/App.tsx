import { Home } from "./pages/home";
import "./App.css";
import QueryProvider from "./context/query-provider";

function App() {
  return (
    <QueryProvider>
      <Home />
    </QueryProvider>
  );
}

export default App;
