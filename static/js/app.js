// static/js/app.js

// Example: Filter symptoms in real-time
document.addEventListener("DOMContentLoaded", () => {
  const searchInput = document.querySelector("#symptomSearch");
  const symptomBoxes = document.querySelectorAll(".symptom-box");

  if (searchInput) {
    searchInput.addEventListener("input", () => {
      const term = searchInput.value.toLowerCase();
      symptomBoxes.forEach((box) => {
        box.style.display = box.innerText.toLowerCase().includes(term)
          ? "flex"
          : "none";
      });
    });
  }
});
