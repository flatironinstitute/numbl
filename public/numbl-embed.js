/*
Define a custom HTML element <numbl-embed> that embeds a numbl iframe.

Usage:
<numbl-embed>
<iframe width="100%" height="500" frameborder="0"></iframe>

<script type="text/plain" class="matlab-script">
... MATLAB script ...
</script>

</numbl-embed>

You can also specify the MATLAB script URL via attribute:
<numbl-embed script="relative-or-absolute-url-to-matlab-script">
<iframe width="100%" height="500" frameborder="0"></iframe>
</numbl-embed>
*/
class NumblEmbed extends HTMLElement {
  constructor() {
    super();
  }

  connectedCallback() {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", () => this.initialize());
    } else {
      this.initialize();
    }
  }

  initialize() {
    this.iframe = this.querySelector("iframe");
    if (!this.iframe) {
      console.error("Missing iframe element in numbl-embed");
      return;
    }

    const defaultNumblUrl = "https://numbl.org";
    const numblUrl = this.attributes["numbl-url"]?.value || defaultNumblUrl;
    const cacheBust = `_cb=${Date.now()}`;
    const mode = this.getAttribute("mode");

    // REPL mode: no script needed, just load the REPL page
    if (mode === "repl") {
      this.iframe.src = `${numblUrl}/embed-repl?${cacheBust}`;
      return;
    }

    const encodePlain = text => {
      const text2 = text.replace(/&lt;/g, "<").replace(/&gt;/g, ">");
      return btoa(text2);
    };

    const scriptElement = this.querySelector("script.matlab-script");

    let scriptBase64;
    if (scriptElement) {
      const matlabScript = scriptElement.textContent.trim();
      scriptBase64 = encodePlain(matlabScript);
    } else if (this.attributes.script) {
      const scriptUrl = this.attributes.script.value;
      this.loadScriptFromUrl(scriptUrl);
      return;
    } else {
      scriptBase64 = null;
    }

    if (scriptBase64) {
      this.iframe.src = `${numblUrl}/embed?script=${scriptBase64}&${cacheBust}`;
    } else {
      this.iframe.src = `${numblUrl}/embed?${cacheBust}`;
    }
  }

  async loadScriptFromUrl(url) {
    try {
      let absoluteUrl = url;
      if (url.startsWith("./") || url.startsWith("../")) {
        const baseUrl = window.location.href;
        absoluteUrl = new URL(url, baseUrl).href;
      }

      const response = await fetch(absoluteUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch script: ${response.statusText}`);
      }

      const scriptContent = await response.text();
      const scriptBase64 = btoa(scriptContent);

      const defaultNumblUrl = "https://numbl.org";
      const numblUrl = this.attributes["numbl-url"]?.value || defaultNumblUrl;

      const cacheBust = `_cb=${Date.now()}`;
      this.iframe.src = `${numblUrl}/embed?script=${scriptBase64}&${cacheBust}`;
    } catch (error) {
      console.error("Error loading MATLAB script:", error);
      this.iframe.srcdoc = `
        <html>
          <body style="font-family: Arial; padding: 20px;">
            <h3>Error loading MATLAB script</h3>
            <p>${error.message}</p>
          </body>
        </html>
      `;
    }
  }
}

customElements.define("numbl-embed", NumblEmbed);
