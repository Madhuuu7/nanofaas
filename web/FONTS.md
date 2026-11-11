# Fonts embedded in the console

`web/index.html` carries two typefaces as base64 `woff2` inside its stylesheet
rather than fetching them. The console is demoed offline and on machines with
no network, and a page that needs a CDN to look like itself looks wrong
exactly when it is being shown to someone. Latin subsets only, about 60 KB in
total.

Both are licensed under the SIL Open Font License, Version 1.1, which permits
embedding. Their copyright notices, as required:

- **Atkinson Hyperlegible** — Copyright © 2020 Braille Institute of America,
  Inc. <https://github.com/googlefonts/atkinson-hyperlegible>
  Drawn so that characters cannot be mistaken for one another, which is the
  argument this console makes about its own numbers.
- **Azeret Mono** — Copyright © 2020 The Azeret Project Authors.
  <https://github.com/displaay/Azeret>
  Holds the figure columns.

The licence text is at <https://openfontlicense.org/>. Neither font is
modified; both are subset to Latin by Google Fonts' own serving pipeline.
