export const windowOpen = (
  url: string,
  type?: "_blank" | "_parent" | "_self" | "_top"
) => {
  const a = document.createElement("a");
  a.setAttribute("href", url);
  a.setAttribute("target", type || "_blank");
  a.rel = "noreferrer";
  document.body.appendChild(a);
  if (a.click) {
    a?.click();
  } else {
    try {
      let evt = new Event("click", {
        bubbles: false,
        cancelable: true,
      });
      a.dispatchEvent(evt);
    } catch (error) {
      window.open(url, type || "_blank");
    }
  }
  document.body.removeChild(a);
};
