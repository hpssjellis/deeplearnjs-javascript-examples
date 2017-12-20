


Youtube video instructions at

https://youtu.be/aq4JByHZt9E







[![Instructional video](http://img.youtube.com/vi/aq4JByHZt9E/0.jpg)](https://youtu.be/aq4JByHZt9E)





weird hack to get Intellisense using VS code working for pure javascript embedded in a web page.

VS Code complains a fair bit but it all seems to work


I use the TS definition files that are already in the deeplearnjs project.

VS code needs a preference changed from false to true (just override it in the spot available)
file-->preferences--> settings "javascript.implicitProjectConfig.checkJs": true

Installed the extension
JavaScript and TypeScript IntelliSense

Not sure if any of these steps were needed, but my hack below works

If you use the [starter typescript](https://github.com/PAIR-code/deeplearnjs/tree/master/starter/typescript) folder and run

yarn prep

Add a file called myPage.ts with this code

[see code here](myPage.ts)

Intellisense works great on that page when writing javascript.

Then to see your working webpage change myPage.ts to myPage.html

and comment the line

//import * as dl from 'deeplearn';

You get:

Intellisense for writing deeplearnjs code in Javascript. Not perfect but it works.

Reverse the last two steps to get back to intellisense.
