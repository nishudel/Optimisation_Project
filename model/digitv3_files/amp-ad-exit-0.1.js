(self.AMP=self.AMP||[]).push({n:"amp-ad-exit",v:"2009252320001",f:(function(AMP,_){
'use strict';var k,ca="function"==typeof Object.create?Object.create:function(a){function b(){}b.prototype=a;return new b};function da(a){for(var b=["object"==typeof globalThis&&globalThis,a,"object"==typeof window&&window,"object"==typeof self&&self,"object"==typeof global&&global],c=0;c<b.length;++c){var d=b[c];if(d&&d.Math==Math)return}(function(){throw Error("Cannot find global object");})()}da(this);"function"===typeof Symbol&&Symbol("x");var l;
if("function"==typeof Object.setPrototypeOf)l=Object.setPrototypeOf;else{var m;a:{var ea={a:!0},n={};try{n.__proto__=ea;m=n.a;break a}catch(a){}m=!1}l=m?function(a,b){a.__proto__=b;if(a.__proto__!==b)throw new TypeError(a+" is not extensible");return a}:null}var q=l;
function r(a,b){a.prototype=ca(b.prototype);a.prototype.constructor=a;if(q)q(a,b);else for(var c in b)if("prototype"!=c)if(Object.defineProperties){var d=Object.getOwnPropertyDescriptor(b,c);d&&Object.defineProperty(a,c,d)}else a[c]=b[c];a.T=b.prototype};function t(a,b){var c=b=void 0===b?"":b;try{return decodeURIComponent(a)}catch(d){return c}};var fa=/(?:^[#?]?|&)([^=&]+)(?:=([^&]*))?/g;function u(a){var b=Object.create(null);if(!a)return b;for(var c;c=fa.exec(a);){var d=t(c[1],c[1]),e=c[2]?t(c[2].replace(/\+/g," "),c[2]):"";b[d]=e}return b};var v="";
function ha(){var a=self;if(a.__AMP_MODE)var b=a.__AMP_MODE;else{b=u(a.location.originalHash||a.location.hash);var c=u(a.location.search);v||(v=a.AMP_CONFIG&&a.AMP_CONFIG.v?a.AMP_CONFIG.v:"012009252320001");b={localDev:!1,development:!!(0<=["1","actions","amp","amp4ads","amp4email"].indexOf(b.development)||a.AMP_DEV_MODE),examiner:"2"==b.development,esm:!1,geoOverride:b["amp-geo"],minified:!0,lite:void 0!=c.amp_lite,test:!1,log:b.log,version:"2009252320001",rtvVersion:v};b=a.__AMP_MODE=b}return b};var ia=Object.prototype.toString;var w=self.AMP_CONFIG||{},ja=("string"==typeof w.cdnProxyRegex?new RegExp(w.cdnProxyRegex):w.cdnProxyRegex)||/^https:\/\/([a-zA-Z0-9_-]+\.)?cdn\.ampproject\.org$/;function x(a){if(self.document&&self.document.head&&(!self.location||!ja.test(self.location.origin))){var b=self.document.head.querySelector('meta[name="'+a+'"]');b&&b.getAttribute("content")}}w.cdnUrl||x("runtime-host");w.geoApiUrl||x("amp-geo-api");self.__AMP_LOG=self.__AMP_LOG||{user:null,dev:null,userForEmbed:null};var y=self.__AMP_LOG;function z(){if(!y.user)throw Error("failed to call initLogConstructor");return y.user}function A(){if(y.dev)return y.dev;throw Error("failed to call initLogConstructor");}function B(a,b,c,d){z().assert(a,b,c,d,void 0,void 0,void 0,void 0,void 0,void 0,void 0)};function C(a,b){this.name=a;this.type=b}C.prototype.filter=function(){};C.prototype.onLayoutMeasure=function(){};function ka(){var a,b;this.promise=new Promise(function(c,d){a=c;b=d});this.resolve=a;this.reject=b};function D(){this.N=100;this.D=this.H=0;this.o=Object.create(null)}D.prototype.has=function(a){return!!this.o[a]};D.prototype.get=function(a){var b=this.o[a];if(b)return b.access=++this.D,b.payload};D.prototype.put=function(a,b){this.has(a)||this.H++;this.o[a]={payload:b,access:this.D};if(!(this.H<=this.N)){A().warn("lru-cache","Trimming LRU cache");a=this.o;var c=this.D+1,d;for(d in a){var e=a[d].access;if(e<c){c=e;var f=d}}void 0!==f&&(delete a[f],this.H--)}};(function(a){return a||{}})({c:!0,v:!0,a:!0,ad:!0});var E,F;
function G(a){E||(E=self.document.createElement("a"),F=self.__AMP_URL_CACHE||(self.__AMP_URL_CACHE=new D));var b=F,c=E;if(b&&b.has(a))a=b.get(a);else{c.href=a;c.protocol||(c.href=c.href);var d={href:c.href,protocol:c.protocol,host:c.host,hostname:c.hostname,port:"0"==c.port?"":c.port,pathname:c.pathname,search:c.search,hash:c.hash,origin:null};"/"!==d.pathname[0]&&(d.pathname="/"+d.pathname);if("http:"==d.protocol&&80==d.port||"https:"==d.protocol&&443==d.port)d.port="",d.host=d.hostname;d.origin=
c.origin&&"null"!=c.origin?c.origin:"data:"!=d.protocol&&d.host?d.protocol+"//"+d.host:d.href;b&&b.put(a,d);a=d}return a};function la(a){var b="";try{"localStorage"in a&&(b=a.localStorage.getItem("amp-experiment-toggles"))}catch(e){A().warn("EXPERIMENTS","Failed to retrieve experiments from localStorage.")}var c=b?b.split(/\s*,\s*/g):[];a=Object.create(null);for(var d=0;d<c.length;d++)0!=c[d].length&&("-"==c[d][0]?a[c[d].substr(1)]=!1:a[c[d]]=!0);return a};function ma(a){var b=a.ownerDocument.defaultView,c=H(b),d=b!=c;if(c.__AMP__EXPERIMENT_TOGGLES)var e=c.__AMP__EXPERIMENT_TOGGLES;else{c.__AMP__EXPERIMENT_TOGGLES=Object.create(null);e=c.__AMP__EXPERIMENT_TOGGLES;if(c.AMP_CONFIG)for(var f in c.AMP_CONFIG){var h=c.AMP_CONFIG[f];"number"===typeof h&&0<=h&&1>=h&&(e[f]=Math.random()<h)}if(c.AMP_CONFIG&&Array.isArray(c.AMP_CONFIG["allow-doc-opt-in"])&&0<c.AMP_CONFIG["allow-doc-opt-in"].length&&(f=c.AMP_CONFIG["allow-doc-opt-in"],h=c.document.head.querySelector('meta[name="amp-experiments-opt-in"]'))){h=
h.getAttribute("content").split(",");for(var g=0;g<h.length;g++)-1!=f.indexOf(h[g])&&(e[h[g]]=!0)}Object.assign(e,la(c));if(c.AMP_CONFIG&&Array.isArray(c.AMP_CONFIG["allow-url-opt-in"])&&0<c.AMP_CONFIG["allow-url-opt-in"].length)for(f=c.AMP_CONFIG["allow-url-opt-in"],c=u(c.location.originalHash||c.location.hash),h=0;h<f.length;h++)g=c["e-"+f[h]],"1"==g&&(e[f[h]]=!0),"0"==g&&(e[f[h]]=!1)}var p=!!e["ampdoc-fie"];d&&!p?a=I(b,"url-replace")?J(b,"url-replace"):null:(a=K(a),a=L(a),a=I(a,"url-replace")?
J(a,"url-replace"):null);return a}function H(a){return a.__AMP_TOP||(a.__AMP_TOP=a)}function K(a){if(a.nodeType){var b=(a.ownerDocument||a).defaultView;b=H(b);a=J(b,"ampdoc").getAmpDoc(a)}return a}function L(a){a=K(a);return a.isSingleDoc()?a.win:a}function J(a,b){I(a,b);a=N(a)[b];a.obj||(a.obj=new a.ctor(a.context),a.ctor=null,a.context=null,a.resolve&&a.resolve(a.obj));return a.obj}
function na(a){var b;(b=N(a)["host-exit"])?b.promise?b=b.promise:(J(a,"host-exit"),b=b.promise=Promise.resolve(b.obj)):b=null;var c=b;if(c)return c;a=N(a);a["host-exit"]=oa();return a["host-exit"].promise}function N(a){var b=a.__AMP_SERVICES;b||(b=a.__AMP_SERVICES={});return b}function I(a,b){a=a.__AMP_SERVICES&&a.__AMP_SERVICES[b];return!(!a||!a.ctor&&!a.obj)}
function oa(){var a=new ka,b=a.promise,c=a.resolve;a=a.reject;b.catch(function(){});return{obj:null,promise:b,resolve:c,reject:a,context:null,ctor:null}};/*
 https://mths.be/cssescape v1.5.1 by @mathias | MIT license */
function O(a,b,c){try{var d=a.open(b,c,void 0)}catch(f){A().error("DOM","Failed to open url on target: ",c,f)}if(!(c=d||"_top"==c)){var e;"number"!==typeof e&&(e=0);c=0<e+8?!1:-1!=="".indexOf("noopener",e)}c||a.open(b,"_top")};var P;function pa(a,b){var c=a,d=b;var e=function(a){try{return d(a)}catch(g){throw self.__AMP_REPORT_ERROR(g),g;}};var f=qa();c.addEventListener("message",e,f?void 0:!1);return function(){c&&c.removeEventListener("message",e,f?void 0:!1);e=c=d=null}}function qa(){if(void 0!==P)return P;P=!1;try{var a={get capture(){P=!0}};self.addEventListener("test-options",null,a);self.removeEventListener("test-options",null,a)}catch(b){}return P};function ra(a,b){return pa(a,b)};var Q={bg:"https://tpc.googlesyndication.com/b4a/b4a-runner.html",moat:"https://z.moatads.com/ampanalytics093284/iframe.html"};Object.assign({},Q,{bg:"https://tpc.googlesyndication.com/b4a/experimental/b4a-runner.html"});function sa(a){B("object"==typeof a);if(a.filters){var b=a.filters,c=["clickDelay","clickLocation","inactiveElement"],d;for(d in b)B("object"==typeof b[d],"Filter specification '%s' is malformed",d),B(-1!=c.indexOf(b[d].type),"Supported filters: "+c.join(", "))}else a.filters={};if(a.transport){b=a.transport;for(var e in b)B("beacon"==e||"image"==e,"Unknown transport option: '"+e+"'"),B("boolean"==typeof b[e])}else a.transport={};e=a.targets;B("object"==typeof e,"'targets' must be an object");for(var f in e)ta(f,
e[f],a);return a}function ta(a,b,c){B("string"==typeof b.finalUrl,"finalUrl of target '%s' must be a string",a);b.filters&&b.filters.forEach(function(a){B(c.filters[a],"filter '%s' not defined",a)});if(b.vars){a=/^_[a-zA-Z0-9_-]+$/;for(var d in b.vars)B(a.test(d),"'%s' must match the pattern '%s'",d,a)}}function R(a){return z().assertString(Q[a],"Unknown or invalid vendor "+a+", note that vendor must use transport: iframe")};function S(a,b,c){C.call(this,a,b.type);B("clickDelay"==b.type&&"number"==typeof b.delay&&0<b.delay,"Invalid ClickDelay spec");this.spec=b;this.intervalStart=Date.now();b.startTimingEvent&&(c.performance&&c.performance.timing?void 0==c.performance.timing[b.startTimingEvent]?A().warn("amp-ad-exit","Invalid performance timing event type "+b.startTimingEvent+", falling back to now"):this.intervalStart=c.performance.timing[b.startTimingEvent]:A().warn("amp-ad-exit","Browser does not support performance timing, falling back to now"))}
r(S,C);S.prototype.filter=function(){return Date.now()-this.intervalStart>=this.spec.delay};function T(a){return{type:"clickDelay",delay:1E3,startTimingEvent:a}};function U(a,b,c){C.call(this,a,b.type);B("clickLocation"==b.type&&("undefined"===typeof b.left||"number"===typeof b.left)&&("undefined"===typeof b.right||"number"===typeof b.right)&&("undefined"===typeof b.top||"number"===typeof b.top)&&("undefined"===typeof b.bottom||"number"===typeof b.bottom)&&("undefined"===typeof b.relativeTo||"string"===typeof b.relativeTo),"Invaid ClickLocation spec");this.O=b.left||0;this.P=b.right||0;this.S=b.top||0;this.M=b.bottom||0;this.G=b.relativeTo;this.J=c;this.h=
{top:0,right:0,bottom:0,left:0}}r(U,C);U.prototype.filter=function(a){return a.clientX>=this.h.left&&a.clientX<=this.h.right&&a.clientY>=this.h.top&&a.clientY<=this.h.bottom?!0:!1};
U.prototype.onLayoutMeasure=function(){var a=this;this.J.getVsync().measure(function(){var b=a.J.win;if(a.G){var c=b.document.querySelector(a.G);B(c,"relativeTo element "+a.G+" not found.");var d=c.getBoundingClientRect();a.h.left=d.left;a.h.top=d.top;a.h.bottom=d.bottom;a.h.right=d.right}else a.h.left=0,a.h.top=0,a.h.bottom=b.innerHeight,a.h.right=b.innerWidth;a.h.left+=a.O;a.h.top+=a.S;a.h.right-=a.P;a.h.bottom-=a.M})};function V(a,b){C.call(this,a,b.type);B("inactiveElement"==b.type&&"string"==typeof b.selector,"Invalid InactiveElementspec");this.R=b.selector}r(V,C);V.prototype.filter=function(a){a=a.target;var b=a.matches||a.webkitMatchesSelector||a.mozMatchesSelector||a.msMatchesSelector||a.oMatchesSelector;return!(b&&b.call(a,this.R))};function W(a,b,c){switch(b.type){case "clickDelay":return new S(a,b,c.win);case "clickLocation":return new U(a,b,c);case "inactiveElement":return new V(a,b)}};function ua(a,b){try{a:{var c=(a.ownerDocument||a).defaultView,d=b||H(c);if(c&&c!=d&&H(c)==d)try{var e=c.frameElement;break a}catch(h){}e=null}var f=e.parentElement;if("AMP-AD"==f.nodeName)return String(f.getResourceId())}catch(h){}return null};function X(a){a=AMP.BaseElement.call(this,a)||this;a.I={};a.K={};a.w=[];a.A={beacon:!0,image:!0};a.C={};a.registerAction("exit",a.exit.bind(a));a.registerAction("setVariable",a.setVariable.bind(a),1);a.L={};a.B=null;a.m=null;a.F={};return a}r(X,AMP.BaseElement);k=X.prototype;
k.exit=function(a){var b=this,c=a.args,d=a.event;B("variable"in c!="target"in c,"One and only one of 'target' and 'variable' must be specified");var e;"variable"in c?((e=this.K[c.variable])||(e=c["default"]),B(e,"Variable target not found, variable:'"+c.variable+"', default:'"+c["default"]+"'"),delete c["default"]):e=c.target;var f=this.I[e];B(f,"Exit target not found: '"+e+"'");B(d,"Unexpected null event");d.preventDefault();if(Y(this.w,d)&&Y(f.filters,d)){var h=va(this,c,d,f);f.trackingUrls&&f.trackingUrls.map(h).forEach(function(a){z().fine("amp-ad-exit",
"pinging "+a);b.A.beacon&&b.win.navigator.sendBeacon&&b.win.navigator.sendBeacon(a,"")||!b.A.image||(b.win.document.createElement("img").src=a)});var g=h(f.finalUrl);K(this.getAmpDoc()).getHeadNode().querySelector("script[host-service]")?na(L(this.getAmpDoc())).then(function(a){return a.openUrl(g)}).catch(function(a){A().fine("amp-ad-exit","ExitServiceError - fallback="+a.fallback);a.fallback&&O(b.win,g,"_blank")}):O(this.win,g,f.behaviors&&f.behaviors.clickTarget&&"_top"==f.behaviors.clickTarget?
"_top":"_blank")}};k.setVariable=function(a){a=a.args;B(this.I[a.target],"Exit target not found: '"+a.target+"'");this.K[a.name]=a.target};
function va(a,b,c,d){var e={CLICK_X:function(){return c.clientX},CLICK_Y:function(){return c.clientY}},f=ma(a.element),h={RANDOM:!0,CLICK_X:!0,CLICK_Y:!0};if(d.vars){var g={},p;for(p in d.vars)g.j=p,"_"==g.j[0]&&(g.l=d.vars[g.j],g.l&&(e[g.j]=function(c){return function(){if(c.l.iframeTransportSignal){var d=f.expandStringSync(c.l.iframeTransportSignal,{IFRAME_TRANSPORT_SIGNAL:function(b,c){if(!b||!c)return"";var d=a.L[b];if(d&&c in d)return d[c]}});if(c.l.iframeTransportSignal=="IFRAME_TRANSPORT_SIGNAL"+
d)A().error("amp-ad-exit","Invalid IFRAME_TRANSPORT_SIGNAL format:"+d+" (perhaps there is a space after a comma?)");else if(""!=d)return d}return c.j in b?b[c.j]:c.l.defaultValue}}(g),h[g.j]=!0)),g={l:g.l,j:g.j}}return function(a){return f.expandUrlSync(a,e,h)}}function Y(a,b){return a.every(function(a){var c=a.filter(b);z().info("amp-ad-exit","Filter '"+a.name+"': "+(c?"pass":"fail"));return c})}
k.buildCallback=function(){var a=this;this.element.setAttribute("aria-hidden","true");this.w.push(W("minDelay",T(),this));this.w.push(W("carouselBtns",{type:"inactiveElement",selector:".amp-carousel-button"},this));var b=this.element.children;B(1==b.length,"The tag should contain exactly one <script> child.");b=b[0];B("SCRIPT"==b.tagName&&b.hasAttribute("type")&&"APPLICATION/JSON"==b.getAttribute("type").toUpperCase(),'The amp-ad-exit config should be inside a <script> tag with type="application/json"');
try{var c=sa(JSON.parse(b.textContent));if("[object Object]"===ia.call(c.options)&&"string"===typeof c.options.startTimingEvent){var d=c.options.startTimingEvent;this.w.splice(0,1,W("minDelay",T(c.options.startTimingEvent),this))}for(var e in c.filters){var f=c.filters[e];"clickDelay"==f.type&&(f.startTimingEvent=f.startTimingEvent||d);this.C[e]=W(e,f,this)}for(var h in c.targets){var g=c.targets[h];this.I[h]={finalUrl:g.finalUrl,trackingUrls:g.trackingUrls||[],vars:g.vars||{},filters:(g.filters||
[]).map(function(b){return a.C[b]}).filter(function(a){return a}),behaviors:g.behaviors||{}};for(var p in g.vars)if(g.vars[p].iframeTransportSignal){var M=g.vars[p].iframeTransportSignal.match(/IFRAME_TRANSPORT_SIGNAL\(([^,]+)/);if(M&&!(2>M.length)){var ba=M[1],Z=G(R(ba)).origin;this.F[Z]=this.F[Z]||ba}}}this.A.beacon=!1!==c.transport.beacon;this.A.image=!1!==c.transport.image}catch(aa){throw this.user().error("amp-ad-exit","Invalid JSON config",aa),aa;}wa(this)};k.resumeCallback=function(){wa(this)};
k.unlayoutCallback=function(){this.B&&(this.B(),this.B=null);return AMP.BaseElement.prototype.unlayoutCallback.call(this)};
function wa(a){"inabox"!=ha().runtime&&(a.m=a.m||ua(a.element,H(a.win)),a.m?a.B=ra(a.getAmpDoc().win,function(b){if(a.F[b.origin]){var c=b.data;if("string"==typeof c&&0==c.indexOf("amp-")&&-1!=c.indexOf("{")){var d=c.indexOf("{");try{var e=JSON.parse(c.substr(d))}catch(h){A().error("MESSAGING","Failed to parse message: "+c,h),e=null}}else e=null;var f=e;f&&"iframe-transport-response"==f.type&&(b=b.origin,B(f.message,"Received empty response from 3p analytics frame"),B(f.creativeId,"Received malformed message from 3p analytics frame: creativeId missing"),
B(f.vendor,"Received malformed message from 3p analytics frame: vendor missing"),e=G(R(f.vendor)),B(e&&e.origin==b,"Invalid origin for vendor "+(f.vendor+": "+b)),f.creativeId==a.m&&(a.L[f.vendor]=f.message))}}):z().warn("amp-ad-exit","No friendly parent amp-ad element was found for amp-ad-exit; not in inabox case."))}k.isLayoutSupported=function(){return!0};k.onLayoutMeasure=function(){for(var a in this.C)this.C[a].onLayoutMeasure()};(function(a){a.registerElement("amp-ad-exit",X)})(self.AMP);
})});

//# sourceMappingURL=amp-ad-exit-0.1.js.map
