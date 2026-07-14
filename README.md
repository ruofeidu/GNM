# GNM: Generative aNthropometric Model and Ecosystem

![GNM Teaser Image](assets/readme/gnm_logo.png)

Welcome to the **GNM Ecosystem** repository. GNM - pronounced as genome
(/ˈdʒiː.noʊm/) in reference to the human genome - strives to be the most
accurate and complete 3D parametric human model.

3D Morphable Models (3DMMs) are widely used across computer vision, computer
graphics, and generative AI for representing human geometry and appearance. GNM
introduces a state-of-the-art family of parametric statistical human models and
its associated perception stack.

Our roadmap includes releasing a comprehensive suite of statistical models
complemented by perception and analysis technology. To facilitate early
community research and open development, we are beginning our open-source
release with **GNM Head**, our high-fidelity statistical 3D model of the human
head.

The ecosystem is released under a permissive license suitable for both
non-commercial and commercial applications.


## GNM Ecosystem Packages

Here we list all the available GNM packages:

<table>
  <thead>
    <tr>
      <th align="left" width="110" nowrap>Name</th>
      <th align="left">Description</th>
      <th align="left" width="160" nowrap>Chips</th>
      <th align="center" width="320" nowrap>Teaser</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left" width="110" nowrap>
        <a href="gnm/shape/README.md"><strong>GNM Head</strong></a>
      </td>
      <td align="left">
        Parametric 3D statistical human head and face geometry model providing fine-grained, disentangled control over identity, expressions, and head pose. The model contains controllable internal anatomy including eyeballs, teeth and tongue. Includes multi-framework backend support for <strong>NumPy</strong>, <strong>JAX</strong>, <strong>PyTorch</strong>, and <strong>TensorFlow</strong>, along with semantic parameter sampling.
      </td>
      <td align="left" width="160" nowrap>
        <a href="https://github.com/google/gnm/actions/workflows/ci-shape-linux.yml"><img src="https://github.com/google/gnm/actions/workflows/ci-shape-linux.yml/badge.svg" alt="CI Linux" /></a><br>
        <a href="https://github.com/google/gnm/actions/workflows/ci-shape-macos.yml"><img src="https://github.com/google/gnm/actions/workflows/ci-shape-macos.yml/badge.svg" alt="CI macOS" /></a><br>
        <a href="https://github.com/google/gnm/actions/workflows/ci-shape-windows.yml"><img src="https://github.com/google/gnm/actions/workflows/ci-shape-windows.yml/badge.svg" alt="CI Windows" /></a><br>
        <a href="https://github.com/google/gnm/actions/workflows/lint.yml"><img src="https://github.com/google/gnm/actions/workflows/lint.yml/badge.svg" alt="Lint" /></a>
      </td>
      <td align="center" width="320" nowrap>
        <a href="gnm/shape/assets/readme/teaser_heads_cropped.gif"><img src="gnm/shape/assets/readme/teaser_heads_cropped.gif" alt="GNM Head Teaser" width="150" /></a>
        <a href="gnm/shape/assets/readme/gnm_head_demo.gif"><img src="gnm/shape/assets/readme/gnm_head_demo.gif" alt="GNM Head demo teaser" width="150" /></a>
      </td>
    </tr>
  </tbody>
</table>

## Citation
If you use any part of the GNM Ecosystem in your work, please consider citing
the corresponding package. Relevant bibtex entries can be found in the
individual packages.

## Contributing
We'd love to accept your patches and contributions to this project! See
[CONTRIBUTING.md](CONTRIBUTING.md) for more information on how to get started
and how we handle external contributions.

## License
This project is licensed under the Apache License, Version 2.0. See the
[LICENSE](LICENSE) file for details.
