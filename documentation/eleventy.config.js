// eleventy.config.js
module.exports = function(eleventyConfig) {
  return {
    dir: {
      input: "src",      // Source files
      includes: "_includes", // For layouts, partials
      data: "_data",       // For global data
      output: "_site"      // Where the built site will go
    },
    passthroughFileCopy: true, // Allow passthrough of static assets
    markdownTemplateEngine: "njk", // Use Nunjucks for Markdown files
    htmlTemplateEngine: "njk",     // Use Nunjucks for HTML files
    templateFormats: ["md", "njk", "html"]
  };
};
