// eleventy.config.js
module.exports = function(eleventyConfig) {
  eleventyConfig.addPassthroughCopy("src/static/css"); // Passthrough for CSS
  eleventyConfig.addPassthroughCopy("src/static/images"); // Passthrough for Images

  return {
    dir: {
      input: "src",      // Source files
      includes: "_includes", // For layouts, partials
      data: "_data",       // For global data
      output: "_site"      // Where the built site will go
    },
    passthroughFileCopy: true, // This might be redundant if addPassthroughCopy is used, but often kept.
    markdownTemplateEngine: "njk", // Use Nunjucks for Markdown files
    htmlTemplateEngine: "njk",     // Use Nunjucks for HTML files
    templateFormats: ["md", "njk", "html"]
  };
};
