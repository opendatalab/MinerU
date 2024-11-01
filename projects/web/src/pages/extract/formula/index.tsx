import { Outlet } from "react-router-dom";

const Formula = () => {
  return (
    <div className="relative w-full h-full flex flex-col items-center justify-center ">
      <Outlet />
    </div>
  );
};

export default Formula;
