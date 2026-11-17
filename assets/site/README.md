# Local research folio

The repository-root [index.html](../../index.html) is a buildless landing page hosted at [the live research site](https://jonland82.github.io/blind-field-discovery/). GitHub Pages publishes the root of `master`; `.nojekyll` keeps the files unchanged. You can also open it directly in a browser; no server, package installation or internet connection is required for local viewing.

`folio.css` holds the parchment palette, type, layout and responsive rules. `folio.js` only switches the displayed conditional-expectation thought experiment. Equations use native MathML; no remote math service or script is required. The room-field figure is the retained figure from note 2. The circular field-line illustration is a conceptual inline SVG, not experimental data.

The two locally bundled fonts are converted to WOFF2 from the Google Fonts repository and retain their SIL Open Font License files:

- EB Garamond: [source font](https://github.com/google/fonts/tree/main/ofl/ebgaramond), [included license](fonts/eb-garamond-OFL.txt).
- Pinyon Script: [source font](https://github.com/google/fonts/tree/main/ofl/pinyonscript), [included license](fonts/pinyon-script-OFL.txt).

The page loads its assets locally and links back to [the GitHub repository](https://github.com/jonland82/blind-field-discovery) in the main navigation. Hosting is configured in the repository’s GitHub Pages settings; no custom deployment workflow is needed. Native MathML needs a modern browser. The site uses relative asset and document links so it works beneath the GitHub Pages repository path as well as locally.
