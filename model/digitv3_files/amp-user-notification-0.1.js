(self.AMP=self.AMP||[]).push({n:"amp-user-notification",v:"2009252320001",f:(function(AMP,_){
'use strict';var l,m="function"==typeof Object.create?Object.create:function(a){function b(){}b.prototype=a;return new b};function n(a){for(var b=["object"==typeof globalThis&&globalThis,a,"object"==typeof window&&window,"object"==typeof self&&self,"object"==typeof global&&global],c=0;c<b.length;++c){var d=b[c];if(d&&d.Math==Math)return}(function(){throw Error("Cannot find global object");})()}n(this);"function"===typeof Symbol&&Symbol("x");var p;
if("function"==typeof Object.setPrototypeOf)p=Object.setPrototypeOf;else{var q;a:{var r={a:!0},t={};try{t.__proto__=r;q=t.a;break a}catch(a){}q=!1}p=q?function(a,b){a.__proto__=b;if(a.__proto__!==b)throw new TypeError(a+" is not extensible");return a}:null}var u=p,v;function w(){return v?v:v=Promise.resolve(void 0)};function x(){var a,b;this.promise=new Promise(function(c,d){a=c;b=d});this.resolve=a;this.reject=b};function y(){this.h=0;this.R=w();this.O=function(){};this.P=function(){}}y.prototype.onQueueEmpty=function(a){this.O=a;0==this.h&&a()};y.prototype.onQueueNotEmpty=function(a){this.P=a;0<this.h&&a()};y.prototype.registerUI=function(a){var b=this;0==this.h&&this.P();this.h++;var c=this.R.then(function(){return a().then(function(){b.h--;0==b.h&&b.O()})});return this.R=c};function z(a,b){var c=b=void 0===b?"":b;try{return decodeURIComponent(a)}catch(d){return c}};var aa=/(?:^[#?]?|&)([^=&]+)(?:=([^&]*))?/g;var A=self.AMP_CONFIG||{},ba=("string"==typeof A.cdnProxyRegex?new RegExp(A.cdnProxyRegex):A.cdnProxyRegex)||/^https:\/\/([a-zA-Z0-9_-]+\.)?cdn\.ampproject\.org$/;function B(a){if(self.document&&self.document.head&&(!self.location||!ba.test(self.location.origin))){var b=self.document.head.querySelector('meta[name="'+a+'"]');b&&b.getAttribute("content")}}A.cdnUrl||B("runtime-host");A.geoApiUrl||B("amp-geo-api");function ca(a){for(var b=null,c="",d=0;d<arguments.length;d++){var g=arguments[d];if(g instanceof Error&&!b){b=void 0;var f=Object.getOwnPropertyDescriptor(g,"message");if(f&&f.writable)b=g;else{f=g.stack;var e=Error(g.message);for(b in g)e[b]=g[b];e.stack=f;b=e}}else c&&(c+=" "),c+=g}b?c&&(b.message=c+": "+b.message):b=Error(c);return b}function da(a){var b=ca.apply(null,arguments);setTimeout(function(){self.__AMP_REPORT_ERROR(b);throw b;})}self.__AMP_LOG=self.__AMP_LOG||{user:null,dev:null,userForEmbed:null};
var C=self.__AMP_LOG;function D(){if(!C.user)throw Error("failed to call initLogConstructor");return C.user}function E(){if(C.dev)return C.dev;throw Error("failed to call initLogConstructor");}function F(a,b,c,d,g){return D().assert(a,b,c,d,g,void 0,void 0,void 0,void 0,void 0,void 0)};function G(a){return a||{}};function H(){this.U=100;this.C=this.G=0;this.o=Object.create(null)}H.prototype.has=function(a){return!!this.o[a]};H.prototype.get=function(a){var b=this.o[a];if(b)return b.access=++this.C,b.payload};H.prototype.put=function(a,b){this.has(a)||this.G++;this.o[a]={payload:b,access:this.C};if(!(this.G<=this.U)){E().warn("lru-cache","Trimming LRU cache");a=this.o;var c=this.C+1,d;for(d in a){var g=a[d].access;if(g<c){c=g;var f=d}}void 0!==f&&(delete a[f],this.G--)}};G({c:!0,v:!0,a:!0,ad:!0});var I,J;
function K(a,b){var c=void 0===c?"source":c;F(null!=a,"%s %s must be available",b,c);var d=a;if("string"==typeof d){I||(I=self.document.createElement("a"),J=self.__AMP_URL_CACHE||(self.__AMP_URL_CACHE=new H));var g=J,f=I;if(g&&g.has(d))d=g.get(d);else{f.href=d;f.protocol||(f.href=f.href);var e={href:f.href,protocol:f.protocol,host:f.host,hostname:f.hostname,port:"0"==f.port?"":f.port,pathname:f.pathname,search:f.search,hash:f.hash,origin:null};"/"!==e.pathname[0]&&(e.pathname="/"+e.pathname);if("http:"==
e.protocol&&80==e.port||"https:"==e.protocol&&443==e.port)e.port="",e.host=e.hostname;e.origin=f.origin&&"null"!=f.origin?f.origin:"data:"!=e.protocol&&e.host?e.protocol+"//"+e.host:e.href;g&&g.put(d,e);d=e}}(g="https:"==d.protocol||"localhost"==d.hostname||"127.0.0.1"==d.hostname)||(d=d.hostname,g=d.length-10,g=0<=g&&d.indexOf(".localhost",g)==g);F(g||/^(\/\/)/.test(a),'%s %s must start with "https://" or "//" or be relative and served from either https or from localhost. Invalid value: %s',b,c,
a)};function ea(a){var b="";try{"localStorage"in a&&(b=a.localStorage.getItem("amp-experiment-toggles"))}catch(g){E().warn("EXPERIMENTS","Failed to retrieve experiments from localStorage.")}var c=b?b.split(/\s*,\s*/g):[];a=Object.create(null);for(var d=0;d<c.length;d++)0!=c[d].length&&("-"==c[d][0]?a[c[d].substr(1)]=!1:a[c[d]]=!0);return a};function fa(a){var b=a.ownerDocument.defaultView,c=b.__AMP_TOP||(b.__AMP_TOP=b),d=b!=c;if(c.__AMP__EXPERIMENT_TOGGLES)var g=c.__AMP__EXPERIMENT_TOGGLES;else{c.__AMP__EXPERIMENT_TOGGLES=Object.create(null);g=c.__AMP__EXPERIMENT_TOGGLES;if(c.AMP_CONFIG)for(var f in c.AMP_CONFIG){var e=c.AMP_CONFIG[f];"number"===typeof e&&0<=e&&1>=e&&(g[f]=Math.random()<e)}if(c.AMP_CONFIG&&Array.isArray(c.AMP_CONFIG["allow-doc-opt-in"])&&0<c.AMP_CONFIG["allow-doc-opt-in"].length&&(f=c.AMP_CONFIG["allow-doc-opt-in"],
e=c.document.head.querySelector('meta[name="amp-experiments-opt-in"]'))){e=e.getAttribute("content").split(",");for(var h=0;h<e.length;h++)-1!=f.indexOf(e[h])&&(g[e[h]]=!0)}Object.assign(g,ea(c));if(c.AMP_CONFIG&&Array.isArray(c.AMP_CONFIG["allow-url-opt-in"])&&0<c.AMP_CONFIG["allow-url-opt-in"].length){f=c.AMP_CONFIG["allow-url-opt-in"];e=c.location.originalHash||c.location.hash;c=Object.create(null);if(e)for(var k;k=aa.exec(e);)h=z(k[1],k[1]),k=k[2]?z(k[2].replace(/\+/g," "),k[2]):"",c[h]=k;for(e=
0;e<f.length;e++)h=c["e-"+f[e]],"1"==h&&(g[f[e]]=!0),"0"==h&&(g[f[e]]=!1)}}var ha=!!g["ampdoc-fie"];d&&!ha?a=L(b,"url-replace")?M(b,"url-replace"):null:(a=N(a),a=O(a),a=L(a,"url-replace")?M(a,"url-replace"):null);return a}function P(a,b){a=a.__AMP_TOP||(a.__AMP_TOP=a);return M(a,b)}function N(a){return a.nodeType?P((a.ownerDocument||a).defaultView,"ampdoc").getAmpDoc(a):a}function O(a){a=N(a);return a.isSingleDoc()?a.win:a}
function M(a,b){L(a,b);a=Q(a)[b];a.obj||(a.obj=new a.ctor(a.context),a.ctor=null,a.context=null,a.resolve&&a.resolve(a.obj));return a.obj}function R(a,b){var c=S(a,b);if(c)return c;a=Q(a);a[b]=ia();return a[b].promise}function S(a,b){var c=Q(a)[b];if(c){if(c.promise)return c.promise;M(a,b);return c.promise=Promise.resolve(c.obj)}return null}function Q(a){var b=a.__AMP_SERVICES;b||(b=a.__AMP_SERVICES={});return b}
function L(a,b){a=a.__AMP_SERVICES&&a.__AMP_SERVICES[b];return!(!a||!a.ctor&&!a.obj)}function ia(){var a=new x,b=a.promise,c=a.resolve;a=a.reject;b.catch(function(){});return{obj:null,promise:b,resolve:c,reject:a,context:null,ctor:null}};/*
 https://mths.be/cssescape v1.5.1 by @mathias | MIT license */
function ja(a){var b=S(O(a),"geo");if(b)return b;var c=N(a);return c.waitForBodyOpen().then(function(){var a=c.win;var b=c.win.document.head;if(b){var f={};b=b.querySelectorAll("script[custom-element],script[custom-template]");for(var e=0;e<b.length;e++){var h=b[e];h=h.getAttribute("custom-element")||h.getAttribute("custom-template");f[h]=!0}f=Object.keys(f)}else f=[];return f.includes("amp-geo")?P(a,"extensions").waitForExtension(a,"amp-geo"):w()}).then(function(){return S(O(a),"geo")})};function T(a){a=AMP.BaseElement.call(this,a)||this;a.J=null;a.m=null;var b=new x;a.K=b.promise;a.L=b.resolve;a.l=null;a.F=!1;a.A=null;a.B=null;a.w=null;a.j=null;a.H="";a.I=null;a.T=null;return a}var U=AMP.BaseElement;T.prototype=m(U.prototype);T.prototype.constructor=T;if(u)u(T,U);else for(var V in U)if("prototype"!=V)if(Object.defineProperties){var W=Object.getOwnPropertyDescriptor(U,V);W&&Object.defineProperty(T,V,W)}else T[V]=U[V];T.Y=U.prototype;l=T.prototype;l.isAlwaysFixed=function(){return!0};
l.buildCallback=function(){var a=this,b=this.getAmpDoc();this.T=fa(this.element);this.I=R(O(this.element),"storage");this.m=F(this.element.id,"amp-user-notification should have an id.");this.H="amp-user-notification:"+this.m;this.A=this.element.getAttribute("data-show-if-geo");this.B=this.element.getAttribute("data-show-if-not-geo");(this.j=this.element.getAttribute("data-show-if-href"))&&K(this.j,this.element);F(1>=!!this.j+!!this.A+!!this.B,'Only one "data-show-if-*" attribute allowed');this.A&&
(this.w=X(this,this.A,!0));this.B&&(this.w=X(this,this.B,!1));(this.l=this.element.getAttribute("data-dismiss-href"))&&K(this.l,this.element);this.element.getAttribute("role")||this.element.setAttribute("role","alert");var c=this.element.getAttribute("data-persist-dismissal");this.F="false"!=c&&"no"!=c;this.registerDefaultAction(function(){return a.dismiss(!1)},"dismiss");this.registerAction("optoutOfCid",function(){return ka(a)});R(O(b),"userNotificationManager").then(function(b){b.registerUserNotification(a.m,
a)})};function X(a,b,c){return ja(a.element).then(function(a){F(a,"requires <amp-geo> to use promptIfUnknownForGeoGroup");var d=b.split(/,\s*/).filter(function(b){return 2==a.isInCountryGroup(b)});return!(c?!d.length:d.length)})}
function la(a,b){return a.T.expandUrlAsync(a.j).then(function(c){var d={elementId:a.m,ampUserId:b};var g=[];for(var f in d){var e=d[f];if(null!=e)if(Array.isArray(e))for(var h=0;h<e.length;h++){var k=e[h];g.push(encodeURIComponent(f)+"="+encodeURIComponent(k))}else g.push(encodeURIComponent(f)+"="+encodeURIComponent(e))}g=g.join("&");g?(d=c.split("#",2),f=d[0].split("?",2),g=f[0]+(f[1]?"?"+f[1]+"&"+g:"?"+g),d=g+=d[1]?"#"+d[1]:""):d=c;return d})}
l.V=function(a){var b=this;this.J=a;return la(this,a).then(function(a){return P(b.win,"xhr").fetchJson(a,{credentials:"include"}).then(function(a){return a.json()})})};function ma(a){var b=a.element.getAttribute("enctype")||"application/json;charset=utf-8";P(a.win,"xhr").fetchJson(a.l,{method:"POST",credentials:"include",body:G({elementId:a.m,ampUserId:a.J}),headers:{"Content-Type":b}})}
l.X=function(a){F("boolean"==typeof a.showNotification,'`showNotification` should be a boolean. Got "%s" which is of type %s.',a.showNotification,typeof a.showNotification);a.showNotification||this.L();return Promise.resolve(a.showNotification)};function ka(a){return R(O(a.element),"cid").then(function(a){return a.optOut()}).then(function(){return a.dismiss(!1)},function(b){E().error("amp-user-notification","Failed to opt out of Cid",b);a.dismiss(!0)})}
function na(a){return R(O(a.element),"cid").then(function(b){return b.get({scope:"amp-user-notification",createCookieIfNotPresent:!0},w(),a.K)})}l.shouldShow=function(){var a=this;return this.isDismissed().then(function(b){return b?!1:a.j?na(a).then(a.V.bind(a)).then(a.X.bind(a)):a.w?a.w:!0})};
l.show=function(){var a=this.element,b=!0;void 0===b&&(b=a.hasAttribute("hidden"));b?a.removeAttribute("hidden"):a.setAttribute("hidden","");this.element.classList.add("amp-active");this.getViewport().addToFixedLayer(this.element);return this.K};l.isDismissed=function(){var a=this;return this.F?this.I.then(function(b){return b.get(a.H)}).then(function(a){return!!a},function(a){E().error("amp-user-notification","Failed to read storage",a);return!1}):Promise.resolve(!1)};
l.dismiss=function(a){var b=this;this.element.classList.remove("amp-active");this.element.classList.add("amp-hidden");this.L();this.getViewport().removeFromFixedLayer(this.element);this.F&&!a&&this.I.then(function(a){a.set(b.H,!0)});this.l&&ma(this)};function Y(a){this.ampdoc=a;this.S=Object.create(null);this.D=Object.create(null);this.M=this.ampdoc.whenReady();this.N=Promise.all([this.ampdoc.whenFirstVisible(),this.M]);this.W=R(O(this.ampdoc),"notificationUIManager")}
Y.prototype.get=function(a){var b=this;this.N.then(function(){null==b.ampdoc.getElementById(a)&&D().warn("amp-user-notification","Did not find amp-user-notification element "+a+".")});return Z(this,a).promise};Y.prototype.getNotification=function(a){var b=this;return this.M.then(function(){return b.S[a]})};
Y.prototype.registerUserNotification=function(a,b){var c=this;this.S[a]=b;var d=Z(this,a);return this.N.then(function(){return b.shouldShow()}).then(function(a){if(a)return c.W.then(function(a){return a.registerUI(b.show.bind(b))})}).then(d.resolve.bind(this,b)).catch(da.bind(null,"Notification service failed amp-user-notification",a))};function Z(a,b){if(a.D[b])return a.D[b];var c=new x;return a.D[b]={promise:c.promise,resolve:c.resolve}}
(function(a){a.registerServiceForDoc("userNotificationManager",Y);a.registerServiceForDoc("notificationUIManager",y);a.registerElement("amp-user-notification",T,"amp-user-notification{position:fixed!important;bottom:0;left:0;overflow:hidden!important;visibility:hidden;background:hsla(0,0%,100%,0.7);z-index:1000;width:100%}amp-user-notification.amp-active{visibility:visible}amp-user-notification.amp-hidden{visibility:hidden}\n/*# sourceURL=/extensions/amp-user-notification/0.1/amp-user-notification.css*/")})(self.AMP);
})});

//# sourceMappingURL=amp-user-notification-0.1.js.map