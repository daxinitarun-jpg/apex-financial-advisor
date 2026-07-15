const fs = require('fs');
const jsdom = require("jsdom");
const { JSDOM } = jsdom;
const html = fs.readFileSync('index.html', 'utf8');

const virtualConsole = new jsdom.VirtualConsole();
virtualConsole.on("error", (err) => {
  console.log("Syntax Error:", err);
});
virtualConsole.on("jsdomError", (err) => {
  console.log("JSDOM Error:", err);
});

const dom = new JSDOM(html, { runScripts: "dangerously", virtualConsole });
console.log("JSDOM loaded. If no errors above, syntax is fine.");
