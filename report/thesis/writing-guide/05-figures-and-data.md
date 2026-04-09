# Figures, Data, and Evaluation

## Chart Abuse

"Chart abuse" = using graphs in a misleading way (intentionally or not).

### Common forms:
1. **Zoomed-in Y-axis**: Makes small variations look dramatic. If the Y-axis
   goes from 78% to 100%, a 21% range looks like nearly 100% change.
   - Not always wrong (zooming in on details is valid) but be AWARE of the
     impression it creates.
2. **Zoomed-out + thick lines**: Hides real differences by making lines look
   identical. Can easily hide 5-10% differences.
3. **Gratuitous logarithmic scales**: Great way to make execution time increases
   look insignificant. Reviewers won't be fooled.

### Rules:
- Be honest. Let the reader judge for themselves.
- Discuss what charts actually show, don't let visual impression do the talking.
- Use diagrams wherever possible to explain concepts — good diagrams help
  readers get through your work quickly.
- Don't use microscopic fonts in figures.

## Units of Measurement

- bit = "b", byte = "B"
- kilo = "k" (NOT "K"). So: kB = kilobytes, kb = kilobits
- K = kelvin. So "KB" = kelvin bytes (nonsense).
- Use SI binary prefixes for powers of two: KiB, MiB, GiB
  (NOT KB, MB, GB when you mean 1024-based units).
- Space between number and unit (half-space preferred).

## Evaluation / Experiments

- Think carefully about what to measure and what you want to learn.
- Think about success criteria BEFORE running experiments.
- Explain any surprising results. Unexplained surprises = cluelessness or error.
- Always provide standard deviations alongside averages.
- Give enough detail for reproducibility.
- Be fair: consider worst-case scenarios for your approach and show they
  aren't too bad.
- Don't construct artificial best cases — they will be discounted.
- Be proactive: anticipate reader's concerns and address them head-on.
- Meet both criteria:
  - **Progressive**: demonstrate significant improvement on the target problem.
  - **Conservative**: demonstrate you haven't worsened other aspects.
