(function () {
  "use strict";

  const brandPattern = /(^|[^A-Za-z0-9_./:=@-])(pyaesa)(?![A-Za-z0-9_./:-])/g;

  function makeBrandFragment(text) {
    const fragment = document.createDocumentFragment();
    let position = 0;
    brandPattern.lastIndex = 0;
    for (const match of text.matchAll(brandPattern)) {
      const brandStart = match.index + match[1].length;
      if (brandStart > position) {
        fragment.appendChild(document.createTextNode(text.slice(position, brandStart)));
      }
      fragment.appendChild(makeBrandNode());
      position = brandStart + match[2].length;
    }
    if (position < text.length) {
      fragment.appendChild(document.createTextNode(text.slice(position)));
    }
    return fragment;
  }

  function makeBrandSpan(text, className) {
    const span = document.createElement("span");
    span.className = className;
    span.textContent = text;
    return span;
  }

  function makeBrandNode() {
    const span = document.createElement("span");
    span.className = "pyaesa-brand";
    span.appendChild(makeBrandSpan("py", "pyaesa-brand-py"));
    span.appendChild(makeBrandSpan("aesa", "pyaesa-brand-aesa"));
    return span;
  }

  function replaceExactBrandLiterals(root) {
    for (const code of root.querySelectorAll("code.docutils.literal")) {
      if (code.textContent.trim() !== "pyaesa") {
        continue;
      }
      code.replaceWith(makeBrandFragment("pyaesa"));
    }
  }

  function shouldSkipTextNode(textNode) {
    const parent = textNode.parentElement;
    if (!parent) {
      return true;
    }
    return Boolean(parent.closest("code, pre, .highlight, .docutils.literal"));
  }

  function replaceBrandText(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const textNodes = [];
    while (walker.nextNode()) {
      textNodes.push(walker.currentNode);
    }
    for (const textNode of textNodes) {
      if (shouldSkipTextNode(textNode) || !brandPattern.test(textNode.nodeValue)) {
        brandPattern.lastIndex = 0;
        continue;
      }
      brandPattern.lastIndex = 0;
      textNode.replaceWith(makeBrandFragment(textNode.nodeValue));
    }
  }

  function applyNavigationBranding() {
    const roots = document.querySelectorAll(
      ".wy-nav-side, .wy-nav-top, .wy-breadcrumbs, .rst-footer-buttons"
    );
    for (const root of roots) {
      replaceExactBrandLiterals(root);
      replaceBrandText(root);
    }
  }

  function cloneText(value) {
    return document.createTextNode(value);
  }

  function trimBoundaryWhitespace(nodes) {
    while (nodes.length && nodes[0].nodeType === Node.TEXT_NODE && !nodes[0].nodeValue.trim()) {
      nodes.shift();
    }
    while (
      nodes.length &&
      nodes[nodes.length - 1].nodeType === Node.TEXT_NODE &&
      !nodes[nodes.length - 1].nodeValue.trim()
    ) {
      nodes.pop();
    }
    if (nodes.length && nodes[0].nodeType === Node.TEXT_NODE) {
      nodes[0].nodeValue = nodes[0].nodeValue.replace(/^[\s\u00a0]+/, "");
    }
    if (nodes.length && nodes[nodes.length - 1].nodeType === Node.TEXT_NODE) {
      nodes[nodes.length - 1].nodeValue = nodes[nodes.length - 1].nodeValue.replace(
        /[\s\u00a0]+$/,
        ""
      );
    }
  }

  function appendClonedNodes(parent, nodes) {
    for (const node of nodes) {
      parent.appendChild(node);
    }
  }

  function segmentBulletParagraph(paragraph) {
    const introNodes = [];
    const segments = [];
    let currentNodes = introNodes;

    for (const child of paragraph.childNodes) {
      if (child.nodeType !== Node.TEXT_NODE) {
        currentNodes.push(child.cloneNode(true));
        continue;
      }

      const parts = child.nodeValue.split("•");
      currentNodes.push(cloneText(parts[0]));
      for (let index = 1; index < parts.length; index += 1) {
        const previousPart = parts[index - 1];
        const indentMatch = previousPart.match(/(?:^|\n)([ \u00a0]*)$/);
        const indent = indentMatch ? indentMatch[1].length : 0;
        const segment = { indent, nodes: [cloneText(parts[index])] };
        segments.push(segment);
        currentNodes = segment.nodes;
      }
    }

    trimBoundaryWhitespace(introNodes);
    for (const segment of segments) {
      trimBoundaryWhitespace(segment.nodes);
    }
    return { introNodes, segments: segments.filter((segment) => segment.nodes.length > 0) };
  }

  function makeLiteral(text) {
    const code = document.createElement("code");
    code.className = "docutils literal notranslate";
    const span = document.createElement("span");
    span.className = "pre";
    span.textContent = text;
    code.appendChild(span);
    return code;
  }

  function promoteListKey(item) {
    const walker = document.createTreeWalker(item, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const textNode = walker.currentNode;
      const match = textNode.nodeValue.match(
        /^([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?|True|False)\s*:\s*/
      );
      if (!match) {
        continue;
      }
      const afterKey = textNode.nodeValue.slice(match[0].length);
      const fragment = document.createDocumentFragment();
      fragment.appendChild(makeLiteral(match[1]));
      fragment.appendChild(document.createTextNode(": "));
      if (afterKey) {
        fragment.appendChild(document.createTextNode(afterKey));
      }
      textNode.replaceWith(fragment);
      return;
    }
  }

  function splitBulletParagraph(paragraph) {
    if (!paragraph.textContent.includes("•")) {
      return;
    }

    const { introNodes, segments } = segmentBulletParagraph(paragraph);
    if (!segments.length) {
      return;
    }

    const fragment = document.createDocumentFragment();
    if (introNodes.length) {
      const introParagraph = document.createElement("p");
      introParagraph.className = "pyaesa-argument-intro";
      appendClonedNodes(introParagraph, introNodes);
      fragment.appendChild(introParagraph);
    }

    const rootList = document.createElement("ul");
    rootList.className = "pyaesa-nested-args";
    let lastRootItem = null;
    for (const segment of segments) {
      const item = document.createElement("li");
      appendClonedNodes(item, segment.nodes);
      promoteListKey(item);
      if (segment.indent > 0 && lastRootItem) {
        let nestedList = lastRootItem.querySelector("ul.pyaesa-nested-args-child");
        if (!nestedList) {
          nestedList = document.createElement("ul");
          nestedList.className = "pyaesa-nested-args pyaesa-nested-args-child";
          lastRootItem.appendChild(nestedList);
        }
        nestedList.appendChild(item);
      } else {
        rootList.appendChild(item);
        lastRootItem = item;
      }
    }

    fragment.appendChild(rootList);
    paragraph.replaceWith(fragment);
  }

  function normalizeArgumentChecklists() {
    for (const paragraph of document.querySelectorAll(".rst-content details > table td p")) {
      if (/^(Nested keys|Accepted keys|Nested mode blocks):$/.test(paragraph.textContent.trim())) {
        paragraph.classList.add("pyaesa-nested-heading");
      }
      splitBulletParagraph(paragraph);
    }
  }

  function applyDocumentationEnhancements() {
    normalizeArgumentChecklists();
    applyNavigationBranding();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", applyDocumentationEnhancements);
  } else {
    applyDocumentationEnhancements();
  }
})();
