// A local thought experiment: changing the available inputs changes the target.
// No fitting, external requests, analytics or stored visitor data.
(() => {
  const buttons = document.querySelectorAll("[data-observation]");
  const answers = document.querySelectorAll("[data-answer]");

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const selection = button.dataset.observation;
      buttons.forEach((candidate) => {
        candidate.setAttribute("aria-pressed", String(candidate === button));
      });
      answers.forEach((answer) => {
        answer.hidden = answer.dataset.answer !== selection;
      });
    });
  });
})();
