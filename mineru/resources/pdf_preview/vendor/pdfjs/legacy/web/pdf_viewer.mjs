/**
 * @licstart The following is the entire license notice for the
 * JavaScript code in this page
 *
 * Copyright 2024 Mozilla Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @licend The above is the entire license notice for the
 * JavaScript code in this page
 */

/**
 * pdfjsVersion = 6.3.289
 * pdfjsBuild = 1c8020a7d
 */
/******/ var __webpack_modules__ = ({

/***/ 9306
(module, __unused_webpack_exports, __webpack_require__) {


var isCallable = __webpack_require__(4901);
var tryToString = __webpack_require__(6823);

var $TypeError = TypeError;

// `Assert: IsCallable(argument) is true`
module.exports = function (argument) {
  if (isCallable(argument)) return argument;
  throw new $TypeError(tryToString(argument) + ' is not a function');
};


/***/ },

/***/ 3506
(module, __unused_webpack_exports, __webpack_require__) {


var isPossiblePrototype = __webpack_require__(3925);

var $String = String;
var $TypeError = TypeError;

module.exports = function (argument) {
  if (isPossiblePrototype(argument)) return argument;
  throw new $TypeError("Can't set " + $String(argument) + ' as a prototype');
};


/***/ },

/***/ 7080
(module, __unused_webpack_exports, __webpack_require__) {


var has = (__webpack_require__(4402).has);

// Perform ? RequireInternalSlot(M, [[SetData]])
module.exports = function (it) {
  has(it);
  return it;
};


/***/ },

/***/ 3463
(module) {


var $TypeError = TypeError;

module.exports = function (argument) {
  if (typeof argument == 'string') return argument;
  throw new $TypeError('Argument is not a string');
};


/***/ },

/***/ 4328
(module, __unused_webpack_exports, __webpack_require__) {


var WeakMapHelpers = __webpack_require__(4995);

var weakmap = new WeakMapHelpers.WeakMap();
var set = WeakMapHelpers.set;
var remove = WeakMapHelpers.remove;

module.exports = function (key) {
  set(weakmap, key, 1);
  remove(weakmap, key);
  return key;
};


/***/ },

/***/ 6557
(module, __unused_webpack_exports, __webpack_require__) {


var has = (__webpack_require__(4995).has);

// Perform ? RequireInternalSlot(M, [[WeakMapData]])
module.exports = function (it) {
  has(it);
  return it;
};


/***/ },

/***/ 6469
(module, __unused_webpack_exports, __webpack_require__) {


var wellKnownSymbol = __webpack_require__(8227);
var create = __webpack_require__(2360);
var defineProperty = (__webpack_require__(4913).f);

var UNSCOPABLES = wellKnownSymbol('unscopables');
var ArrayPrototype = Array.prototype;

// Array.prototype[@@unscopables]
// https://tc39.es/ecma262/#sec-array.prototype-@@unscopables
if (ArrayPrototype[UNSCOPABLES] === undefined) {
  defineProperty(ArrayPrototype, UNSCOPABLES, {
    configurable: true,
    value: create(null)
  });
}

// add a key to Array.prototype[@@unscopables]
module.exports = function (key) {
  ArrayPrototype[UNSCOPABLES][key] = true;
};


/***/ },

/***/ 679
(module, __unused_webpack_exports, __webpack_require__) {


var isPrototypeOf = __webpack_require__(1625);

var $TypeError = TypeError;

module.exports = function (it, Prototype) {
  if (isPrototypeOf(Prototype, it)) return it;
  throw new $TypeError('Incorrect invocation');
};


/***/ },

/***/ 8551
(module, __unused_webpack_exports, __webpack_require__) {


var isObject = __webpack_require__(34);

var $String = String;
var $TypeError = TypeError;

// `Assert: Type(argument) is Object`
module.exports = function (argument) {
  if (isObject(argument)) return argument;
  throw new $TypeError($String(argument) + ' is not an object');
};


/***/ },

/***/ 7811
(module) {


// eslint-disable-next-line es/no-typed-arrays -- safe
module.exports = typeof ArrayBuffer != 'undefined' && typeof DataView != 'undefined';


/***/ },

/***/ 4644
(module, __unused_webpack_exports, __webpack_require__) {


var NATIVE_ARRAY_BUFFER = __webpack_require__(7811);
var DESCRIPTORS = __webpack_require__(3724);
var globalThis = __webpack_require__(4576);
var isCallable = __webpack_require__(4901);
var isObject = __webpack_require__(34);
var hasOwn = __webpack_require__(9297);
var classof = __webpack_require__(6955);
var tryToString = __webpack_require__(6823);
var createNonEnumerableProperty = __webpack_require__(6699);
var defineBuiltIn = __webpack_require__(6840);
var defineBuiltInAccessor = __webpack_require__(2106);
var isPrototypeOf = __webpack_require__(1625);
var getPrototypeOf = __webpack_require__(2787);
var setPrototypeOf = __webpack_require__(2967);
var wellKnownSymbol = __webpack_require__(8227);
var uid = __webpack_require__(3392);
var InternalStateModule = __webpack_require__(1181);

var enforceInternalState = InternalStateModule.enforce;
var getInternalState = InternalStateModule.get;
var Int8Array = globalThis.Int8Array;
var Int8ArrayPrototype = Int8Array && Int8Array.prototype;
var Uint8ClampedArray = globalThis.Uint8ClampedArray;
var Uint8ClampedArrayPrototype = Uint8ClampedArray && Uint8ClampedArray.prototype;
var TypedArray = Int8Array && getPrototypeOf(Int8Array);
var TypedArrayPrototype = Int8ArrayPrototype && getPrototypeOf(Int8ArrayPrototype);
var ObjectPrototype = Object.prototype;
var TypeError = globalThis.TypeError;

var TO_STRING_TAG = wellKnownSymbol('toStringTag');
var TYPED_ARRAY_TAG = uid('TYPED_ARRAY_TAG');
var TYPED_ARRAY_CONSTRUCTOR = 'TypedArrayConstructor';
// Fixing native typed arrays in Opera Presto crashes the browser, see #595
var NATIVE_ARRAY_BUFFER_VIEWS = NATIVE_ARRAY_BUFFER && !!setPrototypeOf && classof(globalThis.opera) !== 'Opera';
var TYPED_ARRAY_TAG_REQUIRED = false;
var NAME, Constructor, Prototype;

var TypedArrayConstructorsList = {
  Int8Array: 1,
  Uint8Array: 1,
  Uint8ClampedArray: 1,
  Int16Array: 2,
  Uint16Array: 2,
  Int32Array: 4,
  Uint32Array: 4,
  Float32Array: 4,
  Float64Array: 8
};

var BigIntArrayConstructorsList = {
  BigInt64Array: 8,
  BigUint64Array: 8
};

var isView = function isView(it) {
  if (!isObject(it)) return false;
  var klass = classof(it);
  return klass === 'DataView'
    || hasOwn(TypedArrayConstructorsList, klass)
    || hasOwn(BigIntArrayConstructorsList, klass);
};

var getTypedArrayConstructor = function (it) {
  var proto = getPrototypeOf(it);
  if (!isObject(proto)) return;
  var state = getInternalState(proto);
  return (state && hasOwn(state, TYPED_ARRAY_CONSTRUCTOR)) ? state[TYPED_ARRAY_CONSTRUCTOR] : getTypedArrayConstructor(proto);
};

var isTypedArray = function (it) {
  if (!isObject(it)) return false;
  var klass = classof(it);
  return hasOwn(TypedArrayConstructorsList, klass)
    || hasOwn(BigIntArrayConstructorsList, klass);
};

var aTypedArray = function (it) {
  if (isTypedArray(it)) return it;
  throw new TypeError('Target is not a typed array');
};

var aTypedArrayConstructor = function (C) {
  if (isCallable(C) && (!setPrototypeOf || isPrototypeOf(TypedArray, C))) return C;
  throw new TypeError(tryToString(C) + ' is not a typed array constructor');
};

var exportTypedArrayMethod = function (KEY, property, forced, options) {
  if (!DESCRIPTORS) return;
  if (forced) for (var ARRAY in TypedArrayConstructorsList) {
    var TypedArrayConstructor = globalThis[ARRAY];
    if (TypedArrayConstructor && hasOwn(TypedArrayConstructor.prototype, KEY)) try {
      delete TypedArrayConstructor.prototype[KEY];
    } catch (error) {
      // old WebKit bug - some methods are non-configurable
      try {
        TypedArrayConstructor.prototype[KEY] = property;
      } catch (error2) { /* empty */ }
    }
  }
  if (!TypedArrayPrototype[KEY] || forced) {
    defineBuiltIn(TypedArrayPrototype, KEY, forced ? property
      : NATIVE_ARRAY_BUFFER_VIEWS && Int8ArrayPrototype[KEY] || property, options);
  }
};

var exportTypedArrayStaticMethod = function (KEY, property, forced) {
  var ARRAY, TypedArrayConstructor;
  if (!DESCRIPTORS) return;
  if (setPrototypeOf) {
    if (forced) for (ARRAY in TypedArrayConstructorsList) {
      TypedArrayConstructor = globalThis[ARRAY];
      if (TypedArrayConstructor && hasOwn(TypedArrayConstructor, KEY)) try {
        delete TypedArrayConstructor[KEY];
      } catch (error) { /* empty */ }
    }
    if (!TypedArray[KEY] || forced) {
      // V8 ~ Chrome 49-50 `%TypedArray%` methods are non-writable non-configurable
      try {
        return defineBuiltIn(TypedArray, KEY, forced ? property : NATIVE_ARRAY_BUFFER_VIEWS && TypedArray[KEY] || property);
      } catch (error) { /* empty */ }
    } else return;
  }
  for (ARRAY in TypedArrayConstructorsList) {
    TypedArrayConstructor = globalThis[ARRAY];
    if (TypedArrayConstructor && (!TypedArrayConstructor[KEY] || forced)) {
      defineBuiltIn(TypedArrayConstructor, KEY, property);
    }
  }
};

for (NAME in TypedArrayConstructorsList) {
  Constructor = globalThis[NAME];
  Prototype = Constructor && Constructor.prototype;
  if (Prototype) enforceInternalState(Prototype)[TYPED_ARRAY_CONSTRUCTOR] = Constructor;
  else NATIVE_ARRAY_BUFFER_VIEWS = false;
}

for (NAME in BigIntArrayConstructorsList) {
  Constructor = globalThis[NAME];
  Prototype = Constructor && Constructor.prototype;
  if (Prototype) enforceInternalState(Prototype)[TYPED_ARRAY_CONSTRUCTOR] = Constructor;
}

// WebKit bug - typed arrays constructors prototype is Object.prototype
if (!NATIVE_ARRAY_BUFFER_VIEWS || !isCallable(TypedArray) || TypedArray === Function.prototype) {
  // eslint-disable-next-line no-shadow -- safe
  TypedArray = function TypedArray() {
    throw new TypeError('Incorrect invocation');
  };
  if (NATIVE_ARRAY_BUFFER_VIEWS) for (NAME in TypedArrayConstructorsList) {
    if (globalThis[NAME]) setPrototypeOf(globalThis[NAME], TypedArray);
  }
}

if (!NATIVE_ARRAY_BUFFER_VIEWS || !TypedArrayPrototype || TypedArrayPrototype === ObjectPrototype) {
  TypedArrayPrototype = TypedArray.prototype;
  if (NATIVE_ARRAY_BUFFER_VIEWS) for (NAME in TypedArrayConstructorsList) {
    if (globalThis[NAME]) setPrototypeOf(globalThis[NAME].prototype, TypedArrayPrototype);
  }
}

// WebKit bug - one more object in Uint8ClampedArray prototype chain
if (NATIVE_ARRAY_BUFFER_VIEWS && getPrototypeOf(Uint8ClampedArrayPrototype) !== TypedArrayPrototype) {
  setPrototypeOf(Uint8ClampedArrayPrototype, TypedArrayPrototype);
}

if (DESCRIPTORS && !hasOwn(TypedArrayPrototype, TO_STRING_TAG)) {
  TYPED_ARRAY_TAG_REQUIRED = true;
  defineBuiltInAccessor(TypedArrayPrototype, TO_STRING_TAG, {
    configurable: true,
    get: function () {
      return isObject(this) ? this[TYPED_ARRAY_TAG] : undefined;
    }
  });
  for (NAME in TypedArrayConstructorsList) if (globalThis[NAME]) {
    createNonEnumerableProperty(globalThis[NAME].prototype, TYPED_ARRAY_TAG, NAME);
  }
}

module.exports = {
  NATIVE_ARRAY_BUFFER_VIEWS: NATIVE_ARRAY_BUFFER_VIEWS,
  TYPED_ARRAY_TAG: TYPED_ARRAY_TAG_REQUIRED && TYPED_ARRAY_TAG,
  aTypedArray: aTypedArray,
  aTypedArrayConstructor: aTypedArrayConstructor,
  exportTypedArrayMethod: exportTypedArrayMethod,
  exportTypedArrayStaticMethod: exportTypedArrayStaticMethod,
  getTypedArrayConstructor: getTypedArrayConstructor,
  isView: isView,
  isTypedArray: isTypedArray,
  TypedArray: TypedArray,
  TypedArrayPrototype: TypedArrayPrototype
};


/***/ },

/***/ 9617
(module, __unused_webpack_exports, __webpack_require__) {


var toIndexedObject = __webpack_require__(5397);
var toAbsoluteIndex = __webpack_require__(5610);
var lengthOfArrayLike = __webpack_require__(6198);

// `Array.prototype.{ indexOf, includes }` methods implementation
var createMethod = function (IS_INCLUDES) {
  return function ($this, el, fromIndex) {
    var O = toIndexedObject($this);
    var length = lengthOfArrayLike(O);
    if (length === 0) return !IS_INCLUDES && -1;
    var index = toAbsoluteIndex(fromIndex, length);
    var value;
    // Array#includes uses SameValueZero equality algorithm
    // eslint-disable-next-line no-self-compare -- NaN check
    if (IS_INCLUDES && el !== el) while (length > index) {
      value = O[index++];
      // eslint-disable-next-line no-self-compare -- NaN check
      if (value !== value) return true;
    // Array#indexOf ignores holes, Array#includes - not
    } else for (;length > index; index++) {
      if ((IS_INCLUDES || index in O) && O[index] === el) return IS_INCLUDES || index || 0;
    } return !IS_INCLUDES && -1;
  };
};

module.exports = {
  // `Array.prototype.includes` method
  // https://tc39.es/ecma262/#sec-array.prototype.includes
  includes: createMethod(true),
  // `Array.prototype.indexOf` method
  // https://tc39.es/ecma262/#sec-array.prototype.indexof
  indexOf: createMethod(false)
};


/***/ },

/***/ 4527
(module, __unused_webpack_exports, __webpack_require__) {


var DESCRIPTORS = __webpack_require__(3724);
var isArray = __webpack_require__(4376);

var $TypeError = TypeError;
// eslint-disable-next-line es/no-object-getownpropertydescriptor -- safe
var getOwnPropertyDescriptor = Object.getOwnPropertyDescriptor;

// Safari < 13 does not throw an error in this case
var SILENT_ON_NON_WRITABLE_LENGTH_SET = DESCRIPTORS && !function () {
  // makes no sense without proper strict mode support
  if (this !== undefined) return true;
  try {
    // eslint-disable-next-line es/no-object-defineproperty -- safe
    Object.defineProperty([], 'length', { writable: false }).length = 1;
  } catch (error) {
    return error instanceof TypeError;
  }
}();

module.exports = SILENT_ON_NON_WRITABLE_LENGTH_SET ? function (O, length) {
  if (isArray(O) && !getOwnPropertyDescriptor(O, 'length').writable) {
    throw new $TypeError('Cannot set read only .length');
  } return O.length = length;
} : function (O, length) {
  return O.length = length;
};


/***/ },

/***/ 6319
(module, __unused_webpack_exports, __webpack_require__) {


var anObject = __webpack_require__(8551);
var iteratorClose = __webpack_require__(9539);

// call something on iterator step with safe closing on error
module.exports = function (iterator, fn, value, ENTRIES) {
  try {
    return ENTRIES ? fn(anObject(value)[0], value[1]) : fn(value);
  } catch (error) {
    iteratorClose(iterator, 'throw', error);
  }
};


/***/ },

/***/ 2195
(module, __unused_webpack_exports, __webpack_require__) {


var uncurryThis = __webpack_require__(9504);

var toString = uncurryThis({}.toString);
var stringSlice = uncurryThis(''.slice);

module.exports = function (it) {
  return stringSlice(toString(it), 8, -1);
};


/***/ },

/***/ 6955
(module, __unused_webpack_exports, __webpack_require__) {


var TO_STRING_TAG_SUPPORT = __webpack_require__(2140);
var isCallable = __webpack_require__(4901);
var classofRaw = __webpack_require__(2195);
var wellKnownSymbol = __webpack_require__(8227);

var TO_STRING_TAG = wellKnownSymbol('toStringTag');
var $Object = Object;

// ES3 wrong here
var CORRECT_ARGUMENTS = classofRaw(function () { return arguments; }()) === 'Arguments';

// fallback for IE11 Script Access Denied error
var tryGet = function (it, key) {
  try {
    return it[key];
  } catch (error) { /* empty */ }
};

// getting tag from ES6+ `Object.prototype.toString`
module.exports = TO_STRING_TAG_SUPPORT ? classofRaw : function (it) {
  var O, tag, result;
  return it === undefined ? 'Undefined' : it === null ? 'Null'
    // @@toStringTag case
    : typeof (tag = tryGet(O = $Object(it), TO_STRING_TAG)) == 'string' ? tag
    // builtinTag case
    : CORRECT_ARGUMENTS ? classofRaw(O)
    // ES3 arguments fallback
    : (result = classofRaw(O)) === 'Object' && isCallable(O.callee) ? 'Arguments' : result;
};


/***/ },

/***/ 7740
(module, __unused_webpack_exports, __webpack_require__) {


var hasOwn = __webpack_require__(9297);
var ownKeys = __webpack_require__(5031);
var getOwnPropertyDescriptorModule = __webpack_require__(7347);
var definePropertyModule = __webpack_require__(4913);

module.exports = function (target, source, exceptions) {
  var keys = ownKeys(source);
  var defineProperty = definePropertyModule.f;
  var getOwnPropertyDescriptor = getOwnPropertyDescriptorModule.f;
  for (var i = 0; i < keys.length; i++) {
    var key = keys[i];
    if (!hasOwn(target, key) && !(exceptions && hasOwn(exceptions, key))) {
      defineProperty(target, key, getOwnPropertyDescriptor(source, key));
    }
  }
};


/***/ },

/***/ 2211
(module, __unused_webpack_exports, __webpack_require__) {


var fails = __webpack_require__(9039);

module.exports = !fails(function () {
  function F() { /* empty */ }
  F.prototype.constructor = null;
  // eslint-disable-next-line es/no-object-getprototypeof -- required for testing
  return Object.getPrototypeOf(new F()) !== F.prototype;
});


/***/ },

/***/ 2529
(module) {


// `CreateIterResultObject` abstract operation
// https://tc39.es/ecma262/#sec-createiterresultobject
module.exports = function (value, done) {
  return { value: value, done: done };
};


/***/ },

/***/ 6699
(module, __unused_webpack_exports, __webpack_require__) {


var DESCRIPTORS = __webpack_require__(3724);
var definePropertyModule = __webpack_require__(4913);
var createPropertyDescriptor = __webpack_require__(6980);

module.exports = DESCRIPTORS ? function (object, key, value) {
  return definePropertyModule.f(object, key, createPropertyDescriptor(1, value));
} : function (object, key, value) {
  object[key] = value;
  return object;
};


/***/ },

/***/ 6980
(module) {


module.exports = function (bitmap, value) {
  return {
    enumerable: !(bitmap & 1),
    configurable: !(bitmap & 2),
    writable: !(bitmap & 4),
    value: value
  };
};


/***/ },

/***/ 4659
(module, __unused_webpack_exports, __webpack_require__) {


var DESCRIPTORS = __webpack_require__(3724);
var definePropertyModule = __webpack_require__(4913);
var createPropertyDescriptor = __webpack_require__(6980);

module.exports = function (object, key, value) {
  if (DESCRIPTORS) definePropertyModule.f(object, key, createPropertyDescriptor(0, value));
  else object[key] = value;
};


/***/ },

/***/ 2106
(module, __unused_webpack_exports, __webpack_require__) {


var makeBuiltIn = __webpack_require__(283);
var defineProperty = __webpack_require__(4913);

module.exports = function (target, name, descriptor) {
  if (descriptor.get) makeBuiltIn(descriptor.get, name, { getter: true });
  if (descriptor.set) makeBuiltIn(descriptor.set, name, { setter: true });
  return defineProperty.f(target, name, descriptor);
};


/***/ },

/***/ 6840
(module, __unused_webpack_exports, __webpack_require__) {


var isCallable = __webpack_require__(4901);
var definePropertyModule = __webpack_require__(4913);
var makeBuiltIn = __webpack_require__(283);
var defineGlobalProperty = __webpack_require__(9433);

module.exports = function (O, key, value, options) {
  if (!options) options = {};
  var simple = options.enumerable;
  var name = options.name !== undefined ? options.name : key;
  if (isCallable(value)) makeBuiltIn(value, name, options);
  if (options.global) {
    if (simple) O[key] = value;
    else defineGlobalProperty(key, value);
  } else {
    try {
      if (!options.unsafe) delete O[key];
      else if (O[key]) simple = true;
    } catch (error) { /* empty */ }
    if (simple) O[key] = value;
    else definePropertyModule.f(O, key, {
      value: value,
      enumerable: false,
      configurable: !options.nonConfigurable,
      writable: !options.nonWritable
    });
  } return O;
};


/***/ },

/***/ 6279
(module, __unused_webpack_exports, __webpack_require__) {


var defineBuiltIn = __webpack_require__(6840);

module.exports = function (target, src, options) {
  for (var key in src) defineBuiltIn(target, key, src[key], options);
  return target;
};


/***/ },

/***/ 9433
(module, __unused_webpack_exports, __webpack_require__) {


var globalThis = __webpack_require__(4576);

// eslint-disable-next-line es/no-object-defineproperty -- safe
var defineProperty = Object.defineProperty;

module.exports = function (key, value) {
  try {
    defineProperty(globalThis, key, { value: value, configurable: true, writable: true });
  } catch (error) {
    globalThis[key] = value;
  } return value;
};


/***/ },

/***/ 3724
(module, __unused_webpack_exports, __webpack_require__) {


var fails = __webpack_require__(9039);

// Detect IE8's incomplete defineProperty implementation
module.exports = !fails(function () {
  // eslint-disable-next-line es/no-object-defineproperty -- required for testing
  return Object.defineProperty({}, 1, { get: function () { return 7; } })[1] !== 7;
});


/***/ },

/***/ 4055
(module, __unused_webpack_exports, __webpack_require__) {


var globalThis = __webpack_require__(4576);
var isObject = __webpack_require__(34);

var document = globalThis.document;
// typeof document.createElement is 'object' in old IE
var EXISTS = isObject(document) && isObject(document.createElement);

module.exports = function (it) {
  return EXISTS ? document.createElement(it) : {};
};


/***/ },

/***/ 6837
(module) {


var $TypeError = TypeError;
var MAX_SAFE_INTEGER = 0x1FFFFFFFFFFFFF; // 2 ** 53 - 1 == 9007199254740991

module.exports = function (it) {
  if (it > MAX_SAFE_INTEGER) throw new $TypeError('Maximum allowed index exceeded');
  return it;
};


/***/ },

/***/ 8727
(module) {


// IE8- don't enum bug keys
module.exports = [
  'constructor',
  'hasOwnProperty',
  'isPrototypeOf',
  'propertyIsEnumerable',
  'toLocaleString',
  'toString',
  'valueOf'
];


/***/ },

/***/ 2839
(module, __unused_webpack_exports, __webpack_require__) {


var globalThis = __webpack_require__(4576);

var navigator = globalThis.navigator;
var userAgent = navigator && navigator.userAgent;

module.exports = userAgent ? String(userAgent) : '';


/***/ },

/***/ 9519
(module, __unused_webpack_exports, __webpack_require__) {


var globalThis = __webpack_require__(4576);
var userAgent = __webpack_require__(2839);

var process = globalThis.process;
var Deno = globalThis.Deno;
var versions = process && process.versions || Deno && Deno.version;
var v8 = versions && versions.v8;
var match, version;

if (v8) {
  match = v8.split('.');
  // in old Chrome, versions of V8 isn't V8 = Chrome / 10
  // but their correct versions are not interesting for us
  version = match[0] > 0 && match[0] < 4 ? 1 : +(match[0] + match[1]);
}

// BrowserFS NodeJS `process` polyfill incorrectly set `.v8` to `0.0`
// so check `userAgent` even if `.v8` exists, but 0
if (!version && userAgent) {
  match = userAgent.match(/Edge\/(\d+)/);
  if (!match || match[1] >= 74) {
    match = userAgent.match(/Chrome\/(\d+)/);
    if (match) version = +match[1];
  }
}

module.exports = version;


/***/ },

/***/ 6518
(module, __unused_webpack_exports, __webpack_require__) {


var globalThis = __webpack_require__(4576);
var getOwnPropertyDescriptor = (__webpack_require__(7347).f);
var createNonEnumerableProperty = __webpack_require__(6699);
var defineBuiltIn = __webpack_require__(6840);
var defineGlobalProperty = __webpack_require__(9433);
var copyConstructorProperties = __webpack_require__(7740);
var isForced = __webpack_require__(2796);

/*
  options.target         - name of the target object
  options.global         - target is the global object
  options.stat           - export as static methods of target
  options.proto          - export as prototype methods of target
  options.real           - real prototype method for the `pure` version
  options.forced         - export even if the native feature is available
  options.bind           - bind methods to the target, required for the `pure` version
  options.wrap           - wrap constructors to preventing global pollution, required for the `pure` version
  options.unsafe         - use the simple assignment of property instead of delete + defineProperty
  options.sham           - add a flag to not completely full polyfills
  options.enumerable     - export as enumerable property
  options.dontCallGetSet - prevent calling a getter on target
  options.name           - the .name of the function if it does not match the key
*/
module.exports = function (options, source) {
  var TARGET = options.target;
  var GLOBAL = options.global;
  var STATIC = options.stat;
  var FORCED, target, key, targetProperty, sourceProperty, descriptor;
  if (GLOBAL) {
    target = globalThis;
  } else if (STATIC) {
    target = globalThis[TARGET] || defineGlobalProperty(TARGET, {});
  } else {
    target = globalThis[TARGET] && globalThis[TARGET].prototype;
  }
  if (target) for (key in source) {
    sourceProperty = source[key];
    if (options.dontCallGetSet) {
      descriptor = getOwnPropertyDescriptor(target, key);
      targetProperty = descriptor && descriptor.value;
    } else targetProperty = target[key];
    FORCED = isForced(GLOBAL ? key : TARGET + (STATIC ? '.' : '#') + key, options.forced);
    // contained in target
    if (!FORCED && targetProperty !== undefined) {
      if (typeof sourceProperty == typeof targetProperty) continue;
      copyConstructorProperties(sourceProperty, targetProperty);
    }
    // add a flag to not completely full polyfills
    if (options.sham || (targetProperty && targetProperty.sham)) {
      createNonEnumerableProperty(sourceProperty, 'sham', true);
    }
    defineBuiltIn(target, key, sourceProperty, options);
  }
};


/***/ },

/***/ 9039
(module) {


module.exports = function (exec) {
  try {
    return !!exec();
  } catch (error) {
    return true;
  }
};


/***/ },

/***/ 6080
(module, __unused_webpack_exports, __webpack_require__) {


var uncurryThis = __webpack_require__(7476);
var aCallable = __webpack_require__(9306);
var NATIVE_BIND = __webpack_require__(616);

var bind = uncurryThis(uncurryThis.bind);

// optional / simple context binding
module.exports = function (fn, that) {
  aCallable(fn);
  return that === undefined ? fn : NATIVE_BIND ? bind(fn, that) : function (/* ...args */) {
    return fn.apply(that, arguments);
  };
};


/***/ },

/***/ 616
(module, __unused_webpack_exports, __webpack_require__) {


var fails = __webpack_require__(9039);

module.exports = !fails(function () {
  // eslint-disable-next-line es/no-function-prototype-bind -- safe
  var test = function () { /* empty */ }.bind();
  // eslint-disable-next-line no-prototype-builtins -- safe
  return typeof test != 'function' || test.hasOwnProperty('prototype');
});


/***/ },

/***/ 9565
(module, __unused_webpack_exports, __webpack_require__) {


var NATIVE_BIND = __webpack_require__(616);

var call = Function.prototype.call;
// eslint-disable-next-line es/no-function-prototype-bind -- safe
module.exports = NATIVE_BIND ? call.bind(call) : function () {
  return call.apply(call, arguments);
};


/***/ },

/***/ 350
(module, __unused_webpack_exports, __webpack_require__) {


var DESCRIPTORS = __webpack_require__(3724);
var hasOwn = __webpack_require__(9297);

var FunctionPrototype = Function.prototype;
// eslint-disable-next-line es/no-object-getownpropertydescriptor -- safe
var getDescriptor = DESCRIPTORS && Object.getOwnPropertyDescriptor;

var EXISTS = hasOwn(FunctionPrototype, 'name');
// additional protection from minified / mangled / dropped function names
var PROPER = EXISTS && function something() { /* empty */ }.name === 'something';
var CONFIGURABLE = EXISTS && (!DESCRIPTORS || (DESCRIPTORS && getDescriptor(FunctionPrototype, 'name').configurable));

module.exports = {
  EXISTS: EXISTS,
  PROPER: PROPER,
  CONFIGURABLE: CONFIGURABLE
};


/***/ },

/***/ 6706
(module, __unused_webpack_exports, __webpack_require__) {


var uncurryThis = __webpack_require__(9504);
var aCallable = __webpack_require__(9306);

module.exports = function (object, key, method) {
  try {
    // eslint-disable-next-line es/no-object-getownpropertydescriptor -- safe
    return uncurryThis(aCallable(Object.getOwnPropertyDescriptor(object, key)[method]));
  } catch (error) { /* empty */ }
};


/***/ },

/***/ 7476
(module, __unused_webpack_exports, __webpack_require__) {


var classofRaw = __webpack_require__(2195);
var uncurryThis = __webpack_require__(9504);

module.exports = function (fn) {
  // Nashorn bug:
  //   https://github.com/zloirock/core-js/issues/1128
  //   https://github.com/zloirock/core-js/issues/1130
  if (classofRaw(fn) === 'Function') return uncurryThis(fn);
};


/***/ },

/***/ 9504
(module, __unused_webpack_exports, __webpack_require__) {


var NATIVE_BIND = __webpack_require__(616);

var FunctionPrototype = Function.prototype;
var call = FunctionPrototype.call;
// eslint-disable-next-line es/no-function-prototype-bind -- safe
var uncurryThisWithBind = NATIVE_BIND && FunctionPrototype.bind.bind(call, call);

module.exports = NATIVE_BIND ? uncurryThisWithBind : function (fn) {
  return function () {
    return call.apply(fn, arguments);
  };
};


/***/ },

/***/ 7751
(module, __unused_webpack_exports, __webpack_require__) {


var globalThis = __webpack_require__(4576);
var isCallable = __webpack_require__(4901);

var aFunction = function (argument) {
  return isCallable(argument) ? argument : undefined;
};

module.exports = function (namespace, method) {
  return arguments.length < 2 ? aFunction(globalThis[namespace]) : globalThis[namespace] && globalThis[namespace][method];
};


/***/ },

/***/ 1767
(module) {


// `GetIteratorDirect(obj)` abstract operation
// https://tc39.es/ecma262/#sec-getiteratordirect
module.exports = function (obj) {
  return {
    iterator: obj,
    next: obj.next,
    done: false
  };
};


/***/ },

/***/ 8563
(module, __unused_webpack_exports, __webpack_require__) {


var call = __webpack_require__(9565);
var isCallable = __webpack_require__(4901);
var anObject = __webpack_require__(8551);
var tryToString = __webpack_require__(6823);
var getIteratorMethod = __webpack_require__(3085);

var $TypeError = TypeError;

module.exports = function (argument, usingIterator) {
  var iteratorMethod = arguments.length < 2 ? getIteratorMethod(argument) : usingIterator;
  if (isCallable(iteratorMethod)) return anObject(call(iteratorMethod, argument));
  throw new $TypeError(tryToString(argument) + ' is not iterable');
};


/***/ },

/***/ 3085
(module, __unused_webpack_exports, __webpack_require__) {


var classof = __webpack_require__(2195);
var isNullOrUndefined = __webpack_require__(4117);
var getMethod = __webpack_require__(5966);
var wellKnownSymbol = __webpack_require__(8227);

var ITERATOR = wellKnownSymbol('iterator');
var ArrayPrototype = Array.prototype;

module.exports = function (it) {
  if (!isNullOrUndefined(it)) return getMethod(it, ITERATOR)
    || getMethod(it, '@@iterator')
    || (classof(it) === 'Arguments' ? ArrayPrototype[ITERATOR] : undefined);
};


/***/ },

/***/ 5966
(module, __unused_webpack_exports, __webpack_require__) {


var aCallable = __webpack_require__(9306);
var isNullOrUndefined = __webpack_require__(4117);

// `GetMethod` abstract operation
// https://tc39.es/ecma262/#sec-getmethod
module.exports = function (V, P) {
  var func = V[P];
  return isNullOrUndefined(func) ? undefined : aCallable(func);
};


/***/ },

/***/ 3789
(module, __unused_webpack_exports, __webpack_require__) {


var aCallable = __webpack_require__(9306);
var anObject = __webpack_require__(8551);
var call = __webpack_require__(9565);
var toIntegerOrInfinity = __webpack_require__(1291);
var getIteratorDirect = __webpack_require__(1767);

var INVALID_SIZE = 'Invalid size';
var $RangeError = RangeError;
var $TypeError = TypeError;
var max = Math.max;

var SetRecord = function (set, intSize) {
  this.set = set;
  this.size = max(intSize, 0);
  this.has = aCallable(set.has);
  this.keys = aCallable(set.keys);
};

SetRecord.prototype = {
  getIterator: function () {
    return getIteratorDirect(anObject(call(this.keys, this.set)));
  },
  includes: function (it) {
    return call(this.has, this.set, it);
  }
};

// `GetSetRecord` abstract operation
// https://tc39.es/proposal-set-methods/#sec-getsetrecord
module.exports = function (obj) {
  anObject(obj);
  var numSize = +obj.size;
  // NOTE: If size is undefined, then numSize will be NaN
  // eslint-disable-next-line no-self-compare -- NaN check
  if (numSize !== numSize) throw new $TypeError(INVALID_SIZE);
  var intSize = toIntegerOrInfinity(numSize);
  if (intSize < 0) throw new $RangeError(INVALID_SIZE);
  return new SetRecord(obj, intSize);
};


/***/ },

/***/ 4576
(module) {


var check = function (it) {
  return it && it.Math === Math && it;
};

// https://github.com/zloirock/core-js/issues/86#issuecomment-115759028
module.exports =
  // eslint-disable-next-line es/no-global-this -- safe
  check(typeof globalThis == 'object' && globalThis) ||
  check(typeof window == 'object' && window) ||
  // eslint-disable-next-line no-restricted-globals -- safe
  check(typeof self == 'object' && self) ||
  check(typeof global == 'object' && global) ||
  check(typeof this == 'object' && this) ||
  // eslint-disable-next-line no-new-func -- fallback
  (function () { return this; })() || Function('return this')();


/***/ },

/***/ 9297
(module, __unused_webpack_exports, __webpack_require__) {


var uncurryThis = __webpack_require__(9504);
var toObject = __webpack_require__(8981);

var hasOwnProperty = uncurryThis({}.hasOwnProperty);

// `HasOwnProperty` abstract operation
// https://tc39.es/ecma262/#sec-hasownproperty
// eslint-disable-next-line es/no-object-hasown -- safe
module.exports = Object.hasOwn || function hasOwn(it, key) {
  return hasOwnProperty(toObject(it), key);
};


/***/ },

/***/ 421
(module) {


module.exports = {};


/***/ },

/***/ 397
(module, __unused_webpack_exports, __webpack_require__) {


var getBuiltIn = __webpack_require__(7751);

module.exports = getBuiltIn('document', 'documentElement');


/***/ },

/***/ 5917
(module, __unused_webpack_exports, __webpack_require__) {


var DESCRIPTORS = __webpack_require__(3724);
var fails = __webpack_require__(9039);
var createElement = __webpack_require__(4055);

// Thanks to IE8 for its funny defineProperty
module.exports = !DESCRIPTORS && !fails(function () {
  // eslint-disable-next-line es/no-object-defineproperty -- required for testing
  return Object.defineProperty(createElement('div'), 'a', {
    get: function () { return 7; }
  }).a !== 7;
});


/***/ },

/***/ 7055
(module, __unused_webpack_exports, __webpack_require__) {


var uncurryThis = __webpack_require__(9504);
var fails = __webpack_require__(9039);
var classof = __webpack_require__(2195);

var $Object = Object;
var split = uncurryThis(''.split);

// fallback for non-array-like ES3 and non-enumerable old V8 strings
module.exports = fails(function () {
  // throws an error in rhino, see https://github.com/mozilla/rhino/issues/346
  // eslint-disable-next-line no-prototype-builtins -- safe
  return !$Object('z').propertyIsEnumerable(0);
}) ? function (it) {
  return classof(it) === 'String' ? split(it, '') : $Object(it);
} : $Object;


/***/ },

/***/ 3706
(module, __unused_webpack_exports, __webpack_require__) {


var uncurryThis = __webpack_require__(9504);
var isCallable = __webpack_require__(4901);
var store = __webpack_require__(7629);

var functionToString = uncurryThis(Function.toString);

// this helper broken in `core-js@3.4.1-3.4.4`, so we can't use `shared` helper
if (!isCallable(store.inspectSource)) {
  store.inspectSource = function (it) {
    return functionToString(it);
  };
}

module.exports = store.inspectSource;


/***/ },

/***/ 1181
(module, __unused_webpack_exports, __webpack_require__) {


var NATIVE_WEAK_MAP = __webpack_require__(8622);
var globalThis = __webpack_require__(4576);
var isObject = __webpack_require__(34);
var createNonEnumerableProperty = __webpack_require__(6699);
var hasOwn = __webpack_require__(9297);
var shared = __webpack_require__(7629);
var sharedKey = __webpack_require__(6119);
var hiddenKeys = __webpack_require__(421);

var OBJECT_ALREADY_INITIALIZED = 'Object already initialized';
var TypeError = globalThis.TypeError;
var WeakMap = globalThis.WeakMap;
var set, get, has;

var enforce = function (it) {
  return has(it) ? get(it) : set(it, {});
};

var getterFor = function (TYPE) {
  return function (it) {
    var state;
    if (!isObject(it) || (state = get(it)).type !== TYPE) {
      throw new TypeError('Incompatible receiver, ' + TYPE + ' required');
    } return state;
  };
};

if (NATIVE_WEAK_MAP || shared.state) {
  var store = shared.state || (shared.state = new WeakMap());
  /* eslint-disable no-self-assign -- prototype methods protection */
  store.get = store.get;
  store.has = store.has;
  store.set = store.set;
  /* eslint-enable no-self-assign -- prototype methods protection */
  set = function (it, metadata) {
    if (store.has(it)) throw new TypeError(OBJECT_ALREADY_INITIALIZED);
    metadata.facade = it;
    store.set(it, metadata);
    return metadata;
  };
  get = function (it) {
    return store.get(it) || {};
  };
  has = function (it) {
    return store.has(it);
  };
} else {
  var STATE = sharedKey('state');
  hiddenKeys[STATE] = true;
  set = function (it, metadata) {
    if (hasOwn(it, STATE)) throw new TypeError(OBJECT_ALREADY_INITIALIZED);
    metadata.facade = it;
    createNonEnumerableProperty(it, STATE, metadata);
    return metadata;
  };
  get = function (it) {
    return hasOwn(it, STATE) ? it[STATE] : {};
  };
  has = function (it) {
    return hasOwn(it, STATE);
  };
}

module.exports = {
  set: set,
  get: get,
  has: has,
  enforce: enforce,
  getterFor: getterFor
};


/***/ },

/***/ 4209
(module, __unused_webpack_exports, __webpack_require__) {


var wellKnownSymbol = __webpack_require__(8227);
var Iterators = __webpack_require__(6269);

var ITERATOR = wellKnownSymbol('iterator');
var ArrayPrototype = Array.prototype;

// check on default Array iterator
module.exports = function (it) {
  return it !== undefined && (Iterators.Array === it || ArrayPrototype[ITERATOR] === it);
};


/***/ },

/***/ 4376
(module, __unused_webpack_exports, __webpack_require__) {


var classof = __webpack_require__(2195);

// `IsArray` abstract operation
// https://tc39.es/ecma262/#sec-isarray
// eslint-disable-next-line es/no-array-isarray -- safe
module.exports = Array.isArray || function isArray(argument) {
  return classof(argument) === 'Array';
};


/***/ },

/***/ 1108
(module, __unused_webpack_exports, __webpack_require__) {


var classof = __webpack_require__(6955);

module.exports = function (it) {
  var klass = classof(it);
  return klass === 'BigInt64Array' || klass === 'BigUint64Array';
};


/***/ },

/***/ 4901
(module) {


// https://tc39.es/ecma262/#sec-IsHTMLDDA-internal-slot
var documentAll = typeof document == 'object' && document.all;

// `IsCallable` abstract operation
// https://tc39.es/ecma262/#sec-iscallable
// eslint-disable-next-line unicorn/no-typeof-undefined -- required for testing
module.exports = typeof documentAll == 'undefined' && documentAll !== undefined ? function (argument) {
  return typeof argument == 'function' || argument === documentAll;
} : function (argument) {
  return typeof argument == 'function';
};


/***/ },

/***/ 2796
(module, __unused_webpack_exports, __webpack_require__) {


var fails = __webpack_require__(9039);
var isCallable = __webpack_require__(4901);

var replacement = /#|\.prototype\./;

var isForced = function (feature, detection) {
  var value = data[normalize(feature)];
  return value === POLYFILL ? true
    : value === NATIVE ? false
    : isCallable(detection) ? fails(detection)
    : !!detection;
};

var normalize = isForced.normalize = function (string) {
  return String(string).replace(replacement, '.').toLowerCase();
};

var data = isForced.data = {};
var NATIVE = isForced.NATIVE = 'N';
var POLYFILL = isForced.POLYFILL = 'P';

module.exports = isForced;


/***/ },

/***/ 4117
(module) {


// we can't use just `it == null` since of `document.all` special case
// https://tc39.es/ecma262/#sec-IsHTMLDDA-internal-slot-aec
module.exports = function (it) {
  return it === null || it === undefined;
};


/***/ },

/***/ 34
(module, __unused_webpack_exports, __webpack_require__) {


var isCallable = __webpack_require__(4901);

module.exports = function (it) {
  return typeof it == 'object' ? it !== null : isCallable(it);
};


/***/ },

/***/ 3925
(module, __unused_webpack_exports, __webpack_require__) {


var isObject = __webpack_require__(34);

module.exports = function (argument) {
  return isObject(argument) || argument === null;
};


/***/ },

/***/ 6395
(module) {


module.exports = false;


/***/ },

/***/ 5810
(module, __unused_webpack_exports, __webpack_require__) {


var isObject = __webpack_require__(34);
var getInternalState = (__webpack_require__(1181).get);

module.exports = function isRawJSON(O) {
  if (!isObject(O)) return false;
  var state = getInternalState(O);
  return !!state && state.type === 'RawJSON';
};


/***/ },

/***/ 757
(module, __unused_webpack_exports, __webpack_require__) {


var getBuiltIn = __webpack_require__(7751);
var isCallable = __webpack_require__(4901);
var isPrototypeOf = __webpack_require__(1625);
var USE_SYMBOL_AS_UID = __webpack_require__(7040);

var $Object = Object;

module.exports = USE_SYMBOL_AS_UID ? function (it) {
  return typeof it == 'symbol';
} : function (it) {
  var $Symbol = getBuiltIn('Symbol');
  return isCallable($Symbol) && isPrototypeOf($Symbol.prototype, $Object(it));
};


/***/ },

/***/ 507
(module, __unused_webpack_exports, __webpack_require__) {


var call = __webpack_require__(9565);

module.exports = function (record, fn, ITERATOR_INSTEAD_OF_RECORD) {
  var iterator = ITERATOR_INSTEAD_OF_RECORD ? record : record.iterator;
  var next = record.next;
  var step, result;
  while (!(step = call(next, iterator)).done) {
    result = fn(step.value);
    if (result !== undefined) return result;
  }
};


/***/ },

/***/ 2652
(module, __unused_webpack_exports, __webpack_require__) {


var bind = __webpack_require__(6080);
var call = __webpack_require__(9565);
var anObject = __webpack_require__(8551);
var tryToString = __webpack_require__(6823);
var isArrayIteratorMethod = __webpack_require__(4209);
var lengthOfArrayLike = __webpack_require__(6198);
var isPrototypeOf = __webpack_require__(1625);
var getIterator = __webpack_require__(8563);
var getIteratorMethod = __webpack_require__(3085);
var iteratorClose = __webpack_require__(9539);

var $TypeError = TypeError;

var Result = function (stopped, result) {
  this.stopped = stopped;
  this.result = result;
};

var ResultPrototype = Result.prototype;

module.exports = function (iterable, unboundFunction, options) {
  var that = options && options.that;
  var AS_ENTRIES = !!(options && options.AS_ENTRIES);
  var IS_RECORD = !!(options && options.IS_RECORD);
  var IS_ITERATOR = !!(options && options.IS_ITERATOR);
  var INTERRUPTED = !!(options && options.INTERRUPTED);
  var fn = bind(unboundFunction, that);
  var iterator, iterFn, index, length, result, next, step;

  var stop = function (condition) {
    var $iterator = iterator;
    iterator = undefined;
    if ($iterator) iteratorClose($iterator, 'normal');
    return new Result(true, condition);
  };

  var callFn = function (value) {
    if (AS_ENTRIES) {
      anObject(value);
      return INTERRUPTED ? fn(value[0], value[1], stop) : fn(value[0], value[1]);
    } return INTERRUPTED ? fn(value, stop) : fn(value);
  };

  if (IS_RECORD) {
    iterator = iterable.iterator;
  } else if (IS_ITERATOR) {
    iterator = iterable;
  } else {
    iterFn = getIteratorMethod(iterable);
    if (!iterFn) throw new $TypeError(tryToString(iterable) + ' is not iterable');
    // optimisation for array iterators
    if (isArrayIteratorMethod(iterFn)) {
      for (index = 0, length = lengthOfArrayLike(iterable); length > index; index++) {
        result = callFn(iterable[index]);
        if (result && isPrototypeOf(ResultPrototype, result)) return result;
      } return new Result(false);
    }
    iterator = getIterator(iterable, iterFn);
  }

  next = IS_RECORD ? iterable.next : iterator.next;
  while (!(step = call(next, iterator)).done) {
    // `IteratorValue` errors should propagate without closing the iterator
    var value = step.value;
    try {
      result = callFn(value);
    } catch (error) {
      if (iterator) iteratorClose(iterator, 'throw', error);
      else throw error;
    }
    if (typeof result == 'object' && result && isPrototypeOf(ResultPrototype, result)) return result;
  } return new Result(false);
};


/***/ },

/***/ 6859
(module) {


// release references held by exhausted / closed iterator helpers to allow GC of the source chain
module.exports = function (state) {
  state.iterator = state.next = state.nextHandler = state.mapper = state.predicate = state.inner =
    state.iterables = state.iters = state.openIters = state.padding = state.finishResults = state.buffer = null;
};


/***/ },

/***/ 1385
(module, __unused_webpack_exports, __webpack_require__) {


var iteratorClose = __webpack_require__(9539);

module.exports = function (iters, kind, value) {
  for (var i = iters.length - 1; i >= 0; i--) {
    if (iters[i] === undefined) continue;
    try {
      value = iteratorClose(iters[i].iterator, kind, value);
    } catch (error) {
      kind = 'throw';
      value = error;
    }
  }
  if (kind === 'throw') throw value;
  return value;
};


/***/ },

/***/ 9539
(module, __unused_webpack_exports, __webpack_require__) {


var call = __webpack_require__(9565);
var anObject = __webpack_require__(8551);
var getMethod = __webpack_require__(5966);

module.exports = function (iterator, kind, value) {
  var innerResult, innerError;
  anObject(iterator);
  try {
    innerResult = getMethod(iterator, 'return');
    if (!innerResult) {
      if (kind === 'throw') throw value;
      return value;
    }
    innerResult = call(innerResult, iterator);
  } catch (error) {
    innerError = true;
    innerResult = error;
  }
  if (kind === 'throw') throw value;
  if (innerError) throw innerResult;
  anObject(innerResult);
  return value;
};


/***/ },

/***/ 9462
(module, __unused_webpack_exports, __webpack_require__) {


var call = __webpack_require__(9565);
var create = __webpack_require__(2360);
var createNonEnumerableProperty = __webpack_require__(6699);
var defineBuiltIns = __webpack_require__(6279);
var wellKnownSymbol = __webpack_require__(8227);
var InternalStateModule = __webpack_require__(1181);
var getMethod = __webpack_require__(5966);
var IteratorPrototype = (__webpack_require__(7657).IteratorPrototype);
var createIterResultObject = __webpack_require__(2529);
var iteratorClose = __webpack_require__(9539);
var iteratorCloseAll = __webpack_require__(1385);
var cleanupState = __webpack_require__(6859);

var TO_STRING_TAG = wellKnownSymbol('toStringTag');
var ITERATOR_HELPER = 'IteratorHelper';
var WRAP_FOR_VALID_ITERATOR = 'WrapForValidIterator';
var NORMAL = 'normal';
var THROW = 'throw';
var setInternalState = InternalStateModule.set;

var createIteratorProxyPrototype = function (IS_ITERATOR) {
  var getInternalState = InternalStateModule.getterFor(IS_ITERATOR ? WRAP_FOR_VALID_ITERATOR : ITERATOR_HELPER);

  return defineBuiltIns(create(IteratorPrototype), {
    next: function next() {
      var state = getInternalState(this);
      // for simplification:
      //   for `%WrapForValidIteratorPrototype%.next` or with `state.returnHandlerResult` our `nextHandler` returns `IterResultObject`
      //   for `%IteratorHelperPrototype%.next` - just a value
      if (IS_ITERATOR) return state.nextHandler();
      if (state.done) return createIterResultObject(undefined, true);
      try {
        var result = state.nextHandler();
        if (state.done) cleanupState(state);
        return state.returnHandlerResult ? result : createIterResultObject(result, state.done);
      } catch (error) {
        state.done = true;
        cleanupState(state);
        throw error;
      }
    },
    'return': function () {
      var state = getInternalState(this);
      var iterator = state.iterator;
      var inner = state.inner;
      var openIters = state.openIters;
      var done = state.done;
      state.done = true;
      if (IS_ITERATOR) {
        var returnMethod = getMethod(iterator, 'return');
        return returnMethod ? call(returnMethod, iterator) : createIterResultObject(undefined, true);
      }
      cleanupState(state);
      if (done) return createIterResultObject(undefined, true);
      if (inner) try {
        iteratorClose(inner.iterator, NORMAL);
      } catch (error) {
        return iteratorClose(iterator, THROW, error);
      }
      if (openIters) try {
        iteratorCloseAll(openIters, NORMAL);
      } catch (error) {
        if (iterator) return iteratorClose(iterator, THROW, error);
        throw error;
      }
      if (iterator) iteratorClose(iterator, NORMAL);
      return createIterResultObject(undefined, true);
    }
  });
};

var WrapForValidIteratorPrototype = createIteratorProxyPrototype(true);
var IteratorHelperPrototype = createIteratorProxyPrototype(false);

createNonEnumerableProperty(IteratorHelperPrototype, TO_STRING_TAG, 'Iterator Helper');

module.exports = function (nextHandler, IS_ITERATOR, RETURN_HANDLER_RESULT) {
  var IteratorProxy = function Iterator(record, state) {
    if (state) {
      state.iterator = record.iterator;
      state.next = record.next;
    } else state = record;
    state.type = IS_ITERATOR ? WRAP_FOR_VALID_ITERATOR : ITERATOR_HELPER;
    state.returnHandlerResult = !!RETURN_HANDLER_RESULT;
    state.nextHandler = nextHandler;
    state.counter = 0;
    state.done = false;
    setInternalState(this, state);
  };

  IteratorProxy.prototype = IS_ITERATOR ? WrapForValidIteratorPrototype : IteratorHelperPrototype;

  return IteratorProxy;
};


/***/ },

/***/ 684
(module) {


// Should throw an error on invalid iterator
// https://issues.chromium.org/issues/336839115
module.exports = function (methodName, argument) {
  // eslint-disable-next-line es/no-iterator -- required for testing
  var method = typeof Iterator == 'function' && Iterator.prototype[methodName];
  if (method) try {
    method.call({ next: null }, argument).next();
  } catch (error) {
    return true;
  }
};


/***/ },

/***/ 4549
(module, __unused_webpack_exports, __webpack_require__) {


var globalThis = __webpack_require__(4576);

// https://github.com/tc39/ecma262/pull/3467
module.exports = function (METHOD_NAME, ExpectedError) {
  var Iterator = globalThis.Iterator;
  var IteratorPrototype = Iterator && Iterator.prototype;
  var method = IteratorPrototype && IteratorPrototype[METHOD_NAME];

  var CLOSED = false;

  if (method) try {
    method.call({
      next: function () { return { done: true }; },
      'return': function () { CLOSED = true; }
    }, -1);
  } catch (error) {
    // https://bugs.webkit.org/show_bug.cgi?id=291195
    if (!(error instanceof ExpectedError)) CLOSED = false;
  }

  if (!CLOSED) return method;
};


/***/ },

/***/ 7657
(module, __unused_webpack_exports, __webpack_require__) {


var fails = __webpack_require__(9039);
var isCallable = __webpack_require__(4901);
var isObject = __webpack_require__(34);
var create = __webpack_require__(2360);
var getPrototypeOf = __webpack_require__(2787);
var defineBuiltIn = __webpack_require__(6840);
var wellKnownSymbol = __webpack_require__(8227);
var IS_PURE = __webpack_require__(6395);

var ITERATOR = wellKnownSymbol('iterator');
var BUGGY_SAFARI_ITERATORS = false;

// `%IteratorPrototype%` object
// https://tc39.es/ecma262/#sec-%iteratorprototype%-object
var IteratorPrototype, PrototypeOfArrayIteratorPrototype, arrayIterator;

/* eslint-disable es/no-array-prototype-keys -- safe */
if ([].keys) {
  arrayIterator = [].keys();
  // Safari 8 has buggy iterators w/o `next`
  if (!('next' in arrayIterator)) BUGGY_SAFARI_ITERATORS = true;
  else {
    PrototypeOfArrayIteratorPrototype = getPrototypeOf(getPrototypeOf(arrayIterator));
    if (PrototypeOfArrayIteratorPrototype !== Object.prototype) IteratorPrototype = PrototypeOfArrayIteratorPrototype;
  }
}

var NEW_ITERATOR_PROTOTYPE = !isObject(IteratorPrototype) || fails(function () {
  var test = {};
  // FF44- legacy iterators case
  return IteratorPrototype[ITERATOR].call(test) !== test;
});

if (NEW_ITERATOR_PROTOTYPE) IteratorPrototype = {};
else if (IS_PURE) IteratorPrototype = create(IteratorPrototype);

// `%IteratorPrototype%[@@iterator]()` method
// https://tc39.es/ecma262/#sec-%iteratorprototype%-@@iterator
if (!isCallable(IteratorPrototype[ITERATOR])) {
  defineBuiltIn(IteratorPrototype, ITERATOR, function () {
    return this;
  });
}

module.exports = {
  IteratorPrototype: IteratorPrototype,
  BUGGY_SAFARI_ITERATORS: BUGGY_SAFARI_ITERATORS
};


/***/ },

/***/ 6269
(module) {


module.exports = Object.create ? Object.create(null) : {};


/***/ },

/***/ 6198
(module, __unused_webpack_exports, __webpack_require__) {


var toLength = __webpack_require__(8014);

// `LengthOfArrayLike` abstract operation
// https://tc39.es/ecma262/#sec-lengthofarraylike
module.exports = function (obj) {
  return toLength(obj.length);
};


/***/ },

/***/ 283
(module, __unused_webpack_exports, __webpack_require__) {


var uncurryThis = __webpack_require__(9504);
var fails = __webpack_require__(9039);
var isCallable = __webpack_require__(4901);
var hasOwn = __webpack_require__(9297);
var DESCRIPTORS = __webpack_require__(3724);
var CONFIGURABLE_FUNCTION_NAME = (__webpack_require__(350).CONFIGURABLE);
var inspectSource = __webpack_require__(3706);
var InternalStateModule = __webpack_require__(1181);

var enforceInternalState = InternalStateModule.enforce;
var getInternalState = InternalStateModule.get;
var $String = String;
// eslint-disable-next-line es/no-object-defineproperty -- safe
var defineProperty = Object.defineProperty;
var stringSlice = uncurryThis(''.slice);
var replace = uncurryThis(''.replace);
var join = uncurryThis([].join);

var CONFIGURABLE_LENGTH = DESCRIPTORS && !fails(function () {
  return defineProperty(function () { /* empty */ }, 'length', { value: 8 }).length !== 8;
});

var TEMPLATE = String(String).split('String');

var makeBuiltIn = module.exports = function (value, name, options) {
  if (stringSlice($String(name), 0, 7) === 'Symbol(') {
    name = '[' + replace($String(name), /^Symbol\(([^)]*)\).*$/, '$1') + ']';
  }
  if (options && options.getter) name = 'get ' + name;
  if (options && options.setter) name = 'set ' + name;
  if (!hasOwn(value, 'name') || (CONFIGURABLE_FUNCTION_NAME && value.name !== name)) {
    if (DESCRIPTORS) defineProperty(value, 'name', { value: name, configurable: true });
    else value.name = name;
  }
  if (CONFIGURABLE_LENGTH && options && hasOwn(options, 'arity') && value.length !== options.arity) {
    defineProperty(value, 'length', { value: options.arity });
  }
  try {
    if (options && hasOwn(options, 'constructor') && options.constructor) {
      if (DESCRIPTORS) defineProperty(value, 'prototype', { writable: false });
    // in V8 ~ Chrome 53, prototypes of some methods, like `Array.prototype.values`, are non-writable
    } else if (value.prototype) value.prototype = undefined;
  } catch (error) { /* empty */ }
  var state = enforceInternalState(value);
  if (!hasOwn(state, 'source')) {
    state.source = join(TEMPLATE, typeof name == 'string' ? name : '');
  } return value;
};

// add fake Function#toString for correct work wrapped methods / constructors with methods like LoDash isNative
// eslint-disable-next-line no-extend-native -- required
Function.prototype.toString = makeBuiltIn(function toString() {
  return isCallable(this) && getInternalState(this).source || inspectSource(this);
}, 'toString');


/***/ },

/***/ 2248
(module, __unused_webpack_exports, __webpack_require__) {


var uncurryThis = __webpack_require__(9504);

// eslint-disable-next-line es/no-map -- safe
var MapPrototype = Map.prototype;

module.exports = {
  // eslint-disable-next-line es/no-map -- safe
  Map: Map,
  set: uncurryThis(MapPrototype.set),
  get: uncurryThis(MapPrototype.get),
  has: uncurryThis(MapPrototype.has),
  remove: uncurryThis(MapPrototype['delete']),
  proto: MapPrototype
};


/***/ },

/***/ 741
(module) {


var ceil = Math.ceil;
var floor = Math.floor;

// `Math.trunc` method
// https://tc39.es/ecma262/#sec-math.trunc
// eslint-disable-next-line es/no-math-trunc -- safe
module.exports = Math.trunc || function trunc(x) {
  var n = +x;
  return (n > 0 ? floor : ceil)(n);
};


/***/ },

/***/ 7819
(module, __unused_webpack_exports, __webpack_require__) {


/* eslint-disable es/no-json -- safe */
var fails = __webpack_require__(9039);

module.exports = !fails(function () {
  var unsafeInt = '9007199254740993';
  // eslint-disable-next-line es/no-json-rawjson -- feature detection
  var raw = JSON.rawJSON(unsafeInt);
  // eslint-disable-next-line es/no-json-israwjson -- feature detection
  return !JSON.isRawJSON(raw) || JSON.stringify(raw) !== unsafeInt;
});


/***/ },

/***/ 2360
(module, __unused_webpack_exports, __webpack_require__) {


/* global ActiveXObject -- old IE, WSH */
var anObject = __webpack_require__(8551);
var definePropertiesModule = __webpack_require__(6801);
var enumBugKeys = __webpack_require__(8727);
var hiddenKeys = __webpack_require__(421);
var html = __webpack_require__(397);
var documentCreateElement = __webpack_require__(4055);
var sharedKey = __webpack_require__(6119);

var GT = '>';
var LT = '<';
var PROTOTYPE = 'prototype';
var SCRIPT = 'script';
var IE_PROTO = sharedKey('IE_PROTO');

var EmptyConstructor = function () { /* empty */ };

var scriptTag = function (content) {
  return LT + SCRIPT + GT + content + LT + '/' + SCRIPT + GT;
};

// Create object with fake `null` prototype: use ActiveX Object with cleared prototype
var NullProtoObjectViaActiveX = function (activeXDocument) {
  activeXDocument.write(scriptTag(''));
  activeXDocument.close();
  var temp = activeXDocument.parentWindow.Object;
  // eslint-disable-next-line no-useless-assignment -- avoid memory leak
  activeXDocument = null;
  return temp;
};

// Create object with fake `null` prototype: use iframe Object with cleared prototype
var NullProtoObjectViaIFrame = function () {
  // Thrash, waste and sodomy: IE GC bug
  var iframe = documentCreateElement('iframe');
  var JS = 'java' + SCRIPT + ':';
  var iframeDocument;
  iframe.style.display = 'none';
  html.appendChild(iframe);
  // https://github.com/zloirock/core-js/issues/475
  iframe.src = String(JS);
  iframeDocument = iframe.contentWindow.document;
  iframeDocument.open();
  iframeDocument.write(scriptTag('document.F=Object'));
  iframeDocument.close();
  return iframeDocument.F;
};

// Check for document.domain and active x support
// No need to use active x approach when document.domain is not set
// see https://github.com/es-shims/es5-shim/issues/150
// variation of https://github.com/kitcambridge/es5-shim/commit/4f738ac066346
// avoid IE GC bug
var activeXDocument;
var NullProtoObject = function () {
  try {
    activeXDocument = new ActiveXObject('htmlfile');
  } catch (error) { /* ignore */ }
  NullProtoObject = typeof document != 'undefined'
    ? document.domain && activeXDocument
      ? NullProtoObjectViaActiveX(activeXDocument) // old IE
      : NullProtoObjectViaIFrame()
    : NullProtoObjectViaActiveX(activeXDocument); // WSH
  var length = enumBugKeys.length;
  while (length--) delete NullProtoObject[PROTOTYPE][enumBugKeys[length]];
  return NullProtoObject();
};

hiddenKeys[IE_PROTO] = true;

// `Object.create` method
// https://tc39.es/ecma262/#sec-object.create
// eslint-disable-next-line es/no-object-create -- safe
module.exports = Object.create || function create(O, Properties) {
  var result;
  if (O !== null) {
    EmptyConstructor[PROTOTYPE] = anObject(O);
    result = new EmptyConstructor();
    EmptyConstructor[PROTOTYPE] = null;
    // add "__proto__" for Object.getPrototypeOf polyfill
    result[IE_PROTO] = O;
  } else result = NullProtoObject();
  return Properties === undefined ? result : definePropertiesModule.f(result, Properties);
};


/***/ },

/***/ 6801
(__unused_webpack_module, exports, __webpack_require__) {


var DESCRIPTORS = __webpack_require__(3724);
var V8_PROTOTYPE_DEFINE_BUG = __webpack_require__(8686);
var definePropertyModule = __webpack_require__(4913);
var anObject = __webpack_require__(8551);
var toIndexedObject = __webpack_require__(5397);
var objectKeys = __webpack_require__(1072);

// `Object.defineProperties` method
// https://tc39.es/ecma262/#sec-object.defineproperties
// eslint-disable-next-line es/no-object-defineproperties -- safe
exports.f = DESCRIPTORS && !V8_PROTOTYPE_DEFINE_BUG ? Object.defineProperties : function defineProperties(O, Properties) {
  anObject(O);
  var props = toIndexedObject(Properties);
  var keys = objectKeys(Properties);
  var length = keys.length;
  var index = 0;
  var key;
  while (length > index) definePropertyModule.f(O, key = keys[index++], props[key]);
  return O;
};


/***/ },

/***/ 4913
(__unused_webpack_module, exports, __webpack_require__) {


var DESCRIPTORS = __webpack_require__(3724);
var IE8_DOM_DEFINE = __webpack_require__(5917);
var V8_PROTOTYPE_DEFINE_BUG = __webpack_require__(8686);
var anObject = __webpack_require__(8551);
var toPropertyKey = __webpack_require__(6969);

var $TypeError = TypeError;
// eslint-disable-next-line es/no-object-defineproperty -- safe
var $defineProperty = Object.defineProperty;
// eslint-disable-next-line es/no-object-getownpropertydescriptor -- safe
var $getOwnPropertyDescriptor = Object.getOwnPropertyDescriptor;
var ENUMERABLE = 'enumerable';
var CONFIGURABLE = 'configurable';
var WRITABLE = 'writable';

// `Object.defineProperty` method
// https://tc39.es/ecma262/#sec-object.defineproperty
exports.f = DESCRIPTORS ? V8_PROTOTYPE_DEFINE_BUG ? function defineProperty(O, P, Attributes) {
  anObject(O);
  P = toPropertyKey(P);
  anObject(Attributes);
  if (typeof O === 'function' && P === 'prototype' && 'value' in Attributes && WRITABLE in Attributes && !Attributes[WRITABLE]) {
    var current = $getOwnPropertyDescriptor(O, P);
    if (current && current[WRITABLE]) {
      O[P] = Attributes.value;
      Attributes = {
        configurable: CONFIGURABLE in Attributes ? Attributes[CONFIGURABLE] : current[CONFIGURABLE],
        enumerable: ENUMERABLE in Attributes ? Attributes[ENUMERABLE] : current[ENUMERABLE],
        writable: false
      };
    }
  } return $defineProperty(O, P, Attributes);
} : $defineProperty : function defineProperty(O, P, Attributes) {
  anObject(O);
  P = toPropertyKey(P);
  anObject(Attributes);
  if (IE8_DOM_DEFINE) try {
    return $defineProperty(O, P, Attributes);
  } catch (error) { /* empty */ }
  if ('get' in Attributes || 'set' in Attributes) throw new $TypeError('Accessors not supported');
  if ('value' in Attributes) O[P] = Attributes.value;
  return O;
};


/***/ },

/***/ 7347
(__unused_webpack_module, exports, __webpack_require__) {


var DESCRIPTORS = __webpack_require__(3724);
var call = __webpack_require__(9565);
var propertyIsEnumerableModule = __webpack_require__(8773);
var createPropertyDescriptor = __webpack_require__(6980);
var toIndexedObject = __webpack_require__(5397);
var toPropertyKey = __webpack_require__(6969);
var hasOwn = __webpack_require__(9297);
var IE8_DOM_DEFINE = __webpack_require__(5917);

// eslint-disable-next-line es/no-object-getownpropertydescriptor -- safe
var $getOwnPropertyDescriptor = Object.getOwnPropertyDescriptor;

// `Object.getOwnPropertyDescriptor` method
// https://tc39.es/ecma262/#sec-object.getownpropertydescriptor
exports.f = DESCRIPTORS ? $getOwnPropertyDescriptor : function getOwnPropertyDescriptor(O, P) {
  O = toIndexedObject(O);
  P = toPropertyKey(P);
  if (IE8_DOM_DEFINE) try {
    return $getOwnPropertyDescriptor(O, P);
  } catch (error) { /* empty */ }
  if (hasOwn(O, P)) return createPropertyDescriptor(!call(propertyIsEnumerableModule.f, O, P), O[P]);
};


/***/ },

/***/ 8480
(__unused_webpack_module, exports, __webpack_require__) {


var internalObjectKeys = __webpack_require__(1828);
var enumBugKeys = __webpack_require__(8727);

var hiddenKeys = enumBugKeys.concat('length', 'prototype');

// `Object.getOwnPropertyNames` method
// https://tc39.es/ecma262/#sec-object.getownpropertynames
// eslint-disable-next-line es/no-object-getownpropertynames -- safe
exports.f = Object.getOwnPropertyNames || function getOwnPropertyNames(O) {
  return internalObjectKeys(O, hiddenKeys);
};


/***/ },

/***/ 3717
(__unused_webpack_module, exports) {


// eslint-disable-next-line es/no-object-getownpropertysymbols -- safe
exports.f = Object.getOwnPropertySymbols;


/***/ },

/***/ 2787
(module, __unused_webpack_exports, __webpack_require__) {


var hasOwn = __webpack_require__(9297);
var isCallable = __webpack_require__(4901);
var toObject = __webpack_require__(8981);
var sharedKey = __webpack_require__(6119);
var CORRECT_PROTOTYPE_GETTER = __webpack_require__(2211);

var IE_PROTO = sharedKey('IE_PROTO');
var $Object = Object;
var ObjectPrototype = $Object.prototype;

// `Object.getPrototypeOf` method
// https://tc39.es/ecma262/#sec-object.getprototypeof
// eslint-disable-next-line es/no-object-getprototypeof -- safe
module.exports = CORRECT_PROTOTYPE_GETTER ? $Object.getPrototypeOf : function (O) {
  var object = toObject(O);
  if (hasOwn(object, IE_PROTO)) return object[IE_PROTO];
  var constructor = object.constructor;
  if (isCallable(constructor) && object instanceof constructor) {
    return constructor.prototype;
  } return object instanceof $Object ? ObjectPrototype : null;
};


/***/ },

/***/ 1625
(module, __unused_webpack_exports, __webpack_require__) {


var uncurryThis = __webpack_require__(9504);

module.exports = uncurryThis({}.isPrototypeOf);


/***/ },

/***/ 1828
(module, __unused_webpack_exports, __webpack_require__) {


var uncurryThis = __webpack_require__(9504);
var hasOwn = __webpack_require__(9297);
var toIndexedObject = __webpack_require__(5397);
var indexOf = (__webpack_require__(9617).indexOf);
var hiddenKeys = __webpack_require__(421);

var push = uncurryThis([].push);

module.exports = function (object, names) {
  var O = toIndexedObject(object);
  var i = 0;
  var result = [];
  var key;
  for (key in O) !hasOwn(hiddenKeys, key) && hasOwn(O, key) && push(result, key);
  // Don't enum bug & hidden keys
  while (names.length > i) if (hasOwn(O, key = names[i++])) {
    ~indexOf(result, key) || push(result, key);
  }
  return result;
};


/***/ },

/***/ 1072
(module, __unused_webpack_exports, __webpack_require__) {


var internalObjectKeys = __webpack_require__(1828);
var enumBugKeys = __webpack_require__(8727);

// `Object.keys` method
// https://tc39.es/ecma262/#sec-object.keys
// eslint-disable-next-line es/no-object-keys -- safe
module.exports = Object.keys || function keys(O) {
  return internalObjectKeys(O, enumBugKeys);
};


/***/ },

/***/ 8773
(__unused_webpack_module, exports) {


var $propertyIsEnumerable = {}.propertyIsEnumerable;
// eslint-disable-next-line es/no-object-getownpropertydescriptor -- safe
var getOwnPropertyDescriptor = Object.getOwnPropertyDescriptor;

// Nashorn ~ JDK8 bug
var NASHORN_BUG = getOwnPropertyDescriptor && !$propertyIsEnumerable.call({ 1: 2 }, 1);

// `Object.prototype.propertyIsEnumerable` method implementation
// https://tc39.es/ecma262/#sec-object.prototype.propertyisenumerable
exports.f = NASHORN_BUG ? function propertyIsEnumerable(V) {
  var descriptor = getOwnPropertyDescriptor(this, V);
  return !!descriptor && descriptor.enumerable;
} : $propertyIsEnumerable;


/***/ },

/***/ 2967
(module, __unused_webpack_exports, __webpack_require__) {


/* eslint-disable no-proto -- safe */
var uncurryThisAccessor = __webpack_require__(6706);
var isObject = __webpack_require__(34);
var requireObjectCoercible = __webpack_require__(7750);
var aPossiblePrototype = __webpack_require__(3506);

// `Object.setPrototypeOf` method
// https://tc39.es/ecma262/#sec-object.setprototypeof
// Works with __proto__ only. Old v8 can't work with null proto objects.
// eslint-disable-next-line es/no-object-setprototypeof -- safe
module.exports = Object.setPrototypeOf || ('__proto__' in {} ? function () {
  var CORRECT_SETTER = false;
  var test = {};
  var setter;
  try {
    setter = uncurryThisAccessor(Object.prototype, '__proto__', 'set');
    setter(test, []);
    CORRECT_SETTER = test instanceof Array;
  } catch (error) { /* empty */ }
  return function setPrototypeOf(O, proto) {
    requireObjectCoercible(O);
    aPossiblePrototype(proto);
    if (!isObject(O)) return O;
    if (CORRECT_SETTER) setter(O, proto);
    else O.__proto__ = proto;
    return O;
  };
}() : undefined);


/***/ },

/***/ 4270
(module, __unused_webpack_exports, __webpack_require__) {


var call = __webpack_require__(9565);
var isCallable = __webpack_require__(4901);
var isObject = __webpack_require__(34);

var $TypeError = TypeError;

// `OrdinaryToPrimitive` abstract operation
// https://tc39.es/ecma262/#sec-ordinarytoprimitive
module.exports = function (input, pref) {
  var fn, val;
  if (pref === 'string' && isCallable(fn = input.toString) && !isObject(val = call(fn, input))) return val;
  if (isCallable(fn = input.valueOf) && !isObject(val = call(fn, input))) return val;
  if (pref !== 'string' && isCallable(fn = input.toString) && !isObject(val = call(fn, input))) return val;
  throw new $TypeError("Can't convert object to primitive value");
};


/***/ },

/***/ 5031
(module, __unused_webpack_exports, __webpack_require__) {


var getBuiltIn = __webpack_require__(7751);
var uncurryThis = __webpack_require__(9504);
var getOwnPropertyNamesModule = __webpack_require__(8480);
var getOwnPropertySymbolsModule = __webpack_require__(3717);
var anObject = __webpack_require__(8551);

var concat = uncurryThis([].concat);

// all object keys, includes non-enumerable and symbols
module.exports = getBuiltIn('Reflect', 'ownKeys') || function ownKeys(it) {
  var keys = getOwnPropertyNamesModule.f(anObject(it));
  var getOwnPropertySymbols = getOwnPropertySymbolsModule.f;
  return getOwnPropertySymbols ? concat(keys, getOwnPropertySymbols(it)) : keys;
};


/***/ },

/***/ 8235
(module, __unused_webpack_exports, __webpack_require__) {


var uncurryThis = __webpack_require__(9504);
var hasOwn = __webpack_require__(9297);

var $SyntaxError = SyntaxError;
var $parseInt = parseInt;
var fromCharCode = String.fromCharCode;
var at = uncurryThis(''.charAt);
var slice = uncurryThis(''.slice);
var exec = uncurryThis(/./.exec);

var codePoints = {
  '\\"': '"',
  '\\\\': '\\',
  '\\/': '/',
  '\\b': '\b',
  '\\f': '\f',
  '\\n': '\n',
  '\\r': '\r',
  '\\t': '\t'
};

var IS_4_HEX_DIGITS = /^[\da-f]{4}$/i;
// eslint-disable-next-line regexp/no-control-character -- safe
var IS_C0_CONTROL_CODE = /^[\u0000-\u001F]$/;

module.exports = function (source, i) {
  var unterminated = true;
  var value = '';
  while (i < source.length) {
    var chr = at(source, i);
    if (chr === '\\') {
      var twoChars = slice(source, i, i + 2);
      if (hasOwn(codePoints, twoChars)) {
        value += codePoints[twoChars];
        i += 2;
      } else if (twoChars === '\\u') {
        i += 2;
        var fourHexDigits = slice(source, i, i + 4);
        if (!exec(IS_4_HEX_DIGITS, fourHexDigits)) throw new $SyntaxError('Bad Unicode escape at: ' + i);
        value += fromCharCode($parseInt(fourHexDigits, 16));
        i += 4;
      } else throw new $SyntaxError('Unknown escape sequence: "' + twoChars + '"');
    } else if (chr === '"') {
      unterminated = false;
      i++;
      break;
    } else {
      if (exec(IS_C0_CONTROL_CODE, chr)) throw new $SyntaxError('Bad control character in string literal at: ' + i);
      value += chr;
      i++;
    }
  }
  if (unterminated) throw new $SyntaxError('Unterminated string at: ' + i);
  return { value: value, end: i };
};


/***/ },

/***/ 7750
(module, __unused_webpack_exports, __webpack_require__) {


var isNullOrUndefined = __webpack_require__(4117);

var $TypeError = TypeError;

// `RequireObjectCoercible` abstract operation
// https://tc39.es/ecma262/#sec-requireobjectcoercible
module.exports = function (it) {
  if (isNullOrUndefined(it)) throw new $TypeError("Can't call method on " + it);
  return it;
};


/***/ },

/***/ 9286
(module, __unused_webpack_exports, __webpack_require__) {


var SetHelpers = __webpack_require__(4402);
var iterate = __webpack_require__(8469);

var Set = SetHelpers.Set;
var add = SetHelpers.add;

module.exports = function (set) {
  var result = new Set();
  iterate(set, function (it) {
    add(result, it);
  });
  return result;
};


/***/ },

/***/ 3440
(module, __unused_webpack_exports, __webpack_require__) {


var aSet = __webpack_require__(7080);
var SetHelpers = __webpack_require__(4402);
var clone = __webpack_require__(9286);
var size = __webpack_require__(5170);
var getSetRecord = __webpack_require__(3789);
var iterateSet = __webpack_require__(8469);
var iterateSimple = __webpack_require__(507);

var has = SetHelpers.has;
var remove = SetHelpers.remove;

// `Set.prototype.difference` method
// https://tc39.es/ecma262/#sec-set.prototype.difference
module.exports = function difference(other) {
  var O = aSet(this);
  var otherRec = getSetRecord(other);
  var result = clone(O);
  if (size(result) <= otherRec.size) iterateSet(result, function (e) {
    if (otherRec.includes(e)) remove(result, e);
  });
  else iterateSimple(otherRec.getIterator(), function (e) {
    if (has(result, e)) remove(result, e);
  });
  return result;
};


/***/ },

/***/ 4402
(module, __unused_webpack_exports, __webpack_require__) {


var uncurryThis = __webpack_require__(9504);

// eslint-disable-next-line es/no-set -- safe
var SetPrototype = Set.prototype;

module.exports = {
  // eslint-disable-next-line es/no-set -- safe
  Set: Set,
  add: uncurryThis(SetPrototype.add),
  has: uncurryThis(SetPrototype.has),
  remove: uncurryThis(SetPrototype['delete']),
  proto: SetPrototype
};


/***/ },

/***/ 8750
(module, __unused_webpack_exports, __webpack_require__) {


var aSet = __webpack_require__(7080);
var SetHelpers = __webpack_require__(4402);
var size = __webpack_require__(5170);
var getSetRecord = __webpack_require__(3789);
var iterateSet = __webpack_require__(8469);
var iterateSimple = __webpack_require__(507);

var Set = SetHelpers.Set;
var add = SetHelpers.add;
var has = SetHelpers.has;

// `Set.prototype.intersection` method
// https://tc39.es/ecma262/#sec-set.prototype.intersection
module.exports = function intersection(other) {
  var O = aSet(this);
  var otherRec = getSetRecord(other);
  var result = new Set();

  if (size(O) > otherRec.size) {
    iterateSimple(otherRec.getIterator(), function (e) {
      if (has(O, e)) add(result, e);
    });
  } else {
    iterateSet(O, function (e) {
      if (otherRec.includes(e)) add(result, e);
    });
  }

  return result;
};


/***/ },

/***/ 4449
(module, __unused_webpack_exports, __webpack_require__) {


var aSet = __webpack_require__(7080);
var has = (__webpack_require__(4402).has);
var size = __webpack_require__(5170);
var getSetRecord = __webpack_require__(3789);
var iterateSet = __webpack_require__(8469);
var iterateSimple = __webpack_require__(507);
var iteratorClose = __webpack_require__(9539);

// `Set.prototype.isDisjointFrom` method
// https://tc39.es/ecma262/#sec-set.prototype.isdisjointfrom
module.exports = function isDisjointFrom(other) {
  var O = aSet(this);
  var otherRec = getSetRecord(other);
  if (size(O) <= otherRec.size) return iterateSet(O, function (e) {
    if (otherRec.includes(e)) return false;
  }, true) !== false;
  var iterator = otherRec.getIterator();
  return iterateSimple(iterator, function (e) {
    if (has(O, e)) return iteratorClose(iterator.iterator, 'normal', false);
  }) !== false;
};


/***/ },

/***/ 3838
(module, __unused_webpack_exports, __webpack_require__) {


var aSet = __webpack_require__(7080);
var size = __webpack_require__(5170);
var iterate = __webpack_require__(8469);
var getSetRecord = __webpack_require__(3789);

// `Set.prototype.isSubsetOf` method
// https://tc39.es/ecma262/#sec-set.prototype.issubsetof
module.exports = function isSubsetOf(other) {
  var O = aSet(this);
  var otherRec = getSetRecord(other);
  if (size(O) > otherRec.size) return false;
  return iterate(O, function (e) {
    if (!otherRec.includes(e)) return false;
  }, true) !== false;
};


/***/ },

/***/ 8527
(module, __unused_webpack_exports, __webpack_require__) {


var aSet = __webpack_require__(7080);
var has = (__webpack_require__(4402).has);
var size = __webpack_require__(5170);
var getSetRecord = __webpack_require__(3789);
var iterateSimple = __webpack_require__(507);
var iteratorClose = __webpack_require__(9539);

// `Set.prototype.isSupersetOf` method
// https://tc39.es/ecma262/#sec-set.prototype.issupersetof
module.exports = function isSupersetOf(other) {
  var O = aSet(this);
  var otherRec = getSetRecord(other);
  if (size(O) < otherRec.size) return false;
  var iterator = otherRec.getIterator();
  return iterateSimple(iterator, function (e) {
    if (!has(O, e)) return iteratorClose(iterator.iterator, 'normal', false);
  }) !== false;
};


/***/ },

/***/ 8469
(module, __unused_webpack_exports, __webpack_require__) {


var uncurryThis = __webpack_require__(9504);
var iterateSimple = __webpack_require__(507);
var SetHelpers = __webpack_require__(4402);

var Set = SetHelpers.Set;
var SetPrototype = SetHelpers.proto;
var forEach = uncurryThis(SetPrototype.forEach);
var keys = uncurryThis(SetPrototype.keys);
var next = keys(new Set()).next;

module.exports = function (set, fn, interruptible) {
  return interruptible ? iterateSimple({ iterator: keys(set), next: next }, fn) : forEach(set, fn);
};


/***/ },

/***/ 4916
(module, __unused_webpack_exports, __webpack_require__) {


var getBuiltIn = __webpack_require__(7751);

var createSetLike = function (size) {
  return {
    size: size,
    has: function () {
      return false;
    },
    keys: function () {
      return {
        next: function () {
          return { done: true };
        }
      };
    }
  };
};

var createSetLikeWithInfinitySize = function (size) {
  return {
    size: size,
    has: function () {
      return true;
    },
    keys: function () {
      throw new Error('e');
    }
  };
};

module.exports = function (name, callback) {
  var Set = getBuiltIn('Set');
  try {
    new Set()[name](createSetLike(0));
    try {
      // late spec change, early WebKit ~ Safari 17 implementation does not pass it
      // https://github.com/tc39/proposal-set-methods/pull/88
      // also covered engines with
      // https://bugs.webkit.org/show_bug.cgi?id=272679
      new Set()[name](createSetLike(-1));
      return false;
    } catch (error2) {
      if (!callback) return true;
      // early V8 implementation bug
      // https://issues.chromium.org/issues/351332634
      try {
        new Set()[name](createSetLikeWithInfinitySize(-Infinity));
        return false;
      } catch (error) {
        var set = new Set([1, 2]);
        return callback(set[name](createSetLikeWithInfinitySize(Infinity)));
      }
    }
  } catch (error) {
    return false;
  }
};


/***/ },

/***/ 9835
(module) {


// Should get iterator record of a set-like object before cloning this
// https://bugs.webkit.org/show_bug.cgi?id=289430
module.exports = function (METHOD_NAME) {
  try {
    // eslint-disable-next-line es/no-set -- needed for test
    var baseSet = new Set();
    var setLike = {
      size: 0,
      has: function () { return true; },
      keys: function () {
        // eslint-disable-next-line es/no-object-defineproperty -- needed for test
        return Object.defineProperty({}, 'next', {
          get: function () {
            baseSet.clear();
            baseSet.add(4);
            return function () {
              return { done: true };
            };
          }
        });
      }
    };
    var result = baseSet[METHOD_NAME](setLike);

    return result.size === 1 && result.values().next().value === 4;
  } catch (error) {
    return false;
  }
};


/***/ },

/***/ 5170
(module, __unused_webpack_exports, __webpack_require__) {


var uncurryThisAccessor = __webpack_require__(6706);
var SetHelpers = __webpack_require__(4402);

module.exports = uncurryThisAccessor(SetHelpers.proto, 'size', 'get') || function (set) {
  return set.size;
};


/***/ },

/***/ 3650
(module, __unused_webpack_exports, __webpack_require__) {


var aSet = __webpack_require__(7080);
var SetHelpers = __webpack_require__(4402);
var clone = __webpack_require__(9286);
var getSetRecord = __webpack_require__(3789);
var iterateSimple = __webpack_require__(507);

var add = SetHelpers.add;
var has = SetHelpers.has;
var remove = SetHelpers.remove;

// `Set.prototype.symmetricDifference` method
// https://tc39.es/ecma262/#sec-set.prototype.symmetricdifference
module.exports = function symmetricDifference(other) {
  var O = aSet(this);
  var keysIter = getSetRecord(other).getIterator();
  var result = clone(O);
  iterateSimple(keysIter, function (e) {
    if (has(O, e)) remove(result, e);
    else add(result, e);
  });
  return result;
};


/***/ },

/***/ 4204
(module, __unused_webpack_exports, __webpack_require__) {


var aSet = __webpack_require__(7080);
var add = (__webpack_require__(4402).add);
var clone = __webpack_require__(9286);
var getSetRecord = __webpack_require__(3789);
var iterateSimple = __webpack_require__(507);

// `Set.prototype.union` method
// https://tc39.es/ecma262/#sec-set.prototype.union
module.exports = function union(other) {
  var O = aSet(this);
  var keysIter = getSetRecord(other).getIterator();
  var result = clone(O);
  iterateSimple(keysIter, function (it) {
    add(result, it);
  });
  return result;
};


/***/ },

/***/ 6119
(module, __unused_webpack_exports, __webpack_require__) {


var shared = __webpack_require__(5745);
var uid = __webpack_require__(3392);

var keys = shared('keys');

module.exports = function (key) {
  return keys[key] || (keys[key] = uid(key));
};


/***/ },

/***/ 7629
(module, __unused_webpack_exports, __webpack_require__) {


var IS_PURE = __webpack_require__(6395);
var globalThis = __webpack_require__(4576);
var defineGlobalProperty = __webpack_require__(9433);

var SHARED = '__core-js_shared__';
var store = module.exports = globalThis[SHARED] || defineGlobalProperty(SHARED, {});

(store.versions || (store.versions = [])).push({
  version: '3.50.0',
  mode: IS_PURE ? 'pure' : 'global',
  copyright: '© 2013–2025 Denis Pushkarev (zloirock.ru), 2025–2026 CoreJS Company (core-js.io). All rights reserved.',
  license: 'https://github.com/zloirock/core-js/blob/v3.50.0/LICENSE',
  source: 'https://github.com/zloirock/core-js'
});


/***/ },

/***/ 5745
(module, __unused_webpack_exports, __webpack_require__) {


var store = __webpack_require__(7629);
// eslint-disable-next-line es/no-object-create -- safe
var create = Object.create || Object;

module.exports = function (key, value) {
  return store[key] || (store[key] = value || create(null));
};


/***/ },

/***/ 533
(module, __unused_webpack_exports, __webpack_require__) {


var uncurryThis = __webpack_require__(9504);
var toLength = __webpack_require__(8014);
var toString = __webpack_require__(655);
var $repeat = __webpack_require__(2333);
var requireObjectCoercible = __webpack_require__(7750);

var repeat = uncurryThis($repeat);
var stringSlice = uncurryThis(''.slice);
var ceil = Math.ceil;

// `String.prototype.{ padStart, padEnd }` methods implementation
var createMethod = function (IS_END) {
  return function ($this, maxLength, fillString) {
    var S = toString(requireObjectCoercible($this));
    var intMaxLength = toLength(maxLength);
    var stringLength = S.length;
    if (intMaxLength <= stringLength) return S;
    var fillStr = fillString === undefined ? ' ' : toString(fillString);
    var fillLen, stringFiller;
    if (fillStr === '') return S;
    fillLen = intMaxLength - stringLength;
    stringFiller = repeat(fillStr, ceil(fillLen / fillStr.length));
    if (stringFiller.length > fillLen) stringFiller = stringSlice(stringFiller, 0, fillLen);
    return IS_END ? S + stringFiller : stringFiller + S;
  };
};

module.exports = {
  // `String.prototype.padStart` method
  // https://tc39.es/ecma262/#sec-string.prototype.padstart
  start: createMethod(false),
  // `String.prototype.padEnd` method
  // https://tc39.es/ecma262/#sec-string.prototype.padend
  end: createMethod(true)
};


/***/ },

/***/ 2333
(module, __unused_webpack_exports, __webpack_require__) {


var toIntegerOrInfinity = __webpack_require__(1291);
var toString = __webpack_require__(655);
var requireObjectCoercible = __webpack_require__(7750);

var $RangeError = RangeError;
var floor = Math.floor;

// `String.prototype.repeat` method implementation
// https://tc39.es/ecma262/#sec-string.prototype.repeat
module.exports = function repeat(count) {
  var str = toString(requireObjectCoercible(this));
  var result = '';
  var n = toIntegerOrInfinity(count);
  if (n < 0 || n === Infinity) throw new $RangeError('Wrong number of repetitions');
  for (;n > 0; (n = floor(n / 2)) && (str += str)) if (n % 2) result += str;
  return result;
};


/***/ },

/***/ 4495
(module, __unused_webpack_exports, __webpack_require__) {


/* eslint-disable es/no-symbol -- required for testing */
var V8_VERSION = __webpack_require__(9519);
var fails = __webpack_require__(9039);
var globalThis = __webpack_require__(4576);

var $String = globalThis.String;

// eslint-disable-next-line es/no-object-getownpropertysymbols -- required for testing
module.exports = !!Object.getOwnPropertySymbols && !fails(function () {
  var symbol = Symbol('symbol detection');
  // Chrome 38 Symbol has incorrect toString conversion
  // `get-own-property-symbols` polyfill symbols converted to object are not Symbol instances
  // nb: Do not call `String` directly to avoid this being optimized out to `symbol+''` which will,
  // of course, fail.
  return !$String(symbol) || !(Object(symbol) instanceof Symbol) ||
    // Chrome 38-40 symbols are not inherited from DOM collections prototypes to instances
    !Symbol.sham && V8_VERSION && V8_VERSION < 41;
});


/***/ },

/***/ 1240
(module, __unused_webpack_exports, __webpack_require__) {


var uncurryThis = __webpack_require__(9504);

// `thisNumberValue` abstract operation
// https://tc39.es/ecma262/#sec-thisnumbervalue
module.exports = uncurryThis(1.1.valueOf);


/***/ },

/***/ 5610
(module, __unused_webpack_exports, __webpack_require__) {


var toIntegerOrInfinity = __webpack_require__(1291);

var max = Math.max;
var min = Math.min;

// Helper for a popular repeating case of the spec:
// Let integer be ? ToInteger(index).
// If integer < 0, let result be max((length + integer), 0); else let result be min(integer, length).
module.exports = function (index, length) {
  var integer = toIntegerOrInfinity(index);
  return integer < 0 ? max(integer + length, 0) : min(integer, length);
};


/***/ },

/***/ 5854
(module, __unused_webpack_exports, __webpack_require__) {


var toPrimitive = __webpack_require__(2777);

var $TypeError = TypeError;

// `ToBigInt` abstract operation
// https://tc39.es/ecma262/#sec-tobigint
module.exports = function (argument) {
  var prim = toPrimitive(argument, 'number');
  if (typeof prim == 'number') throw new $TypeError("Can't convert number to bigint");
  // eslint-disable-next-line es/no-bigint -- safe
  return BigInt(prim);
};


/***/ },

/***/ 5397
(module, __unused_webpack_exports, __webpack_require__) {


// toObject with fallback for non-array-like ES3 strings
var IndexedObject = __webpack_require__(7055);
var requireObjectCoercible = __webpack_require__(7750);

module.exports = function (it) {
  return IndexedObject(requireObjectCoercible(it));
};


/***/ },

/***/ 1291
(module, __unused_webpack_exports, __webpack_require__) {


var trunc = __webpack_require__(741);

// `ToIntegerOrInfinity` abstract operation
// https://tc39.es/ecma262/#sec-tointegerorinfinity
module.exports = function (argument) {
  var number = +argument;
  // eslint-disable-next-line no-self-compare -- NaN check
  return number !== number || number === 0 ? 0 : trunc(number);
};


/***/ },

/***/ 8014
(module, __unused_webpack_exports, __webpack_require__) {


var toIntegerOrInfinity = __webpack_require__(1291);

var min = Math.min;

// `ToLength` abstract operation
// https://tc39.es/ecma262/#sec-tolength
module.exports = function (argument) {
  var len = toIntegerOrInfinity(argument);
  return len > 0 ? min(len, 0x1FFFFFFFFFFFFF) : 0; // 2 ** 53 - 1 == 9007199254740991
};


/***/ },

/***/ 8981
(module, __unused_webpack_exports, __webpack_require__) {


var requireObjectCoercible = __webpack_require__(7750);

var $Object = Object;

// `ToObject` abstract operation
// https://tc39.es/ecma262/#sec-toobject
module.exports = function (argument) {
  return $Object(requireObjectCoercible(argument));
};


/***/ },

/***/ 2777
(module, __unused_webpack_exports, __webpack_require__) {


var call = __webpack_require__(9565);
var isObject = __webpack_require__(34);
var isSymbol = __webpack_require__(757);
var getMethod = __webpack_require__(5966);
var ordinaryToPrimitive = __webpack_require__(4270);
var wellKnownSymbol = __webpack_require__(8227);

var $TypeError = TypeError;
var TO_PRIMITIVE = wellKnownSymbol('toPrimitive');

// `ToPrimitive` abstract operation
// https://tc39.es/ecma262/#sec-toprimitive
module.exports = function (input, pref) {
  if (!isObject(input) || isSymbol(input)) return input;
  var exoticToPrim = getMethod(input, TO_PRIMITIVE);
  var result;
  if (exoticToPrim) {
    if (pref === undefined) pref = 'default';
    result = call(exoticToPrim, input, pref);
    if (!isObject(result) || isSymbol(result)) return result;
    throw new $TypeError("Can't convert object to primitive value");
  }
  if (pref === undefined) pref = 'number';
  return ordinaryToPrimitive(input, pref);
};


/***/ },

/***/ 6969
(module, __unused_webpack_exports, __webpack_require__) {


var toPrimitive = __webpack_require__(2777);
var isSymbol = __webpack_require__(757);

// `ToPropertyKey` abstract operation
// https://tc39.es/ecma262/#sec-topropertykey
module.exports = function (argument) {
  var key = toPrimitive(argument, 'string');
  return isSymbol(key) ? key : key + '';
};


/***/ },

/***/ 2140
(module, __unused_webpack_exports, __webpack_require__) {


var wellKnownSymbol = __webpack_require__(8227);

var TO_STRING_TAG = wellKnownSymbol('toStringTag');
var test = {};
// eslint-disable-next-line unicorn/no-immediate-mutation -- ES3 syntax limitation
test[TO_STRING_TAG] = 'z';

module.exports = String(test) === '[object z]';


/***/ },

/***/ 655
(module, __unused_webpack_exports, __webpack_require__) {


var classof = __webpack_require__(6955);

var $String = String;

module.exports = function (argument) {
  if (classof(argument) === 'Symbol') throw new TypeError('Cannot convert a Symbol value to a string');
  return $String(argument);
};


/***/ },

/***/ 6823
(module) {


var $String = String;

module.exports = function (argument) {
  try {
    return $String(argument);
  } catch (error) {
    return 'Object';
  }
};


/***/ },

/***/ 3392
(module, __unused_webpack_exports, __webpack_require__) {


var uncurryThis = __webpack_require__(9504);

var id = 0;
var postfix = Math.random();
var toString = uncurryThis(1.1.toString);

module.exports = function (key) {
  return 'Symbol(' + (key === undefined ? '' : key) + ')_' + toString(++id + postfix, 36);
};


/***/ },

/***/ 7416
(module, __unused_webpack_exports, __webpack_require__) {


var fails = __webpack_require__(9039);
var wellKnownSymbol = __webpack_require__(8227);
var DESCRIPTORS = __webpack_require__(3724);
var IS_PURE = __webpack_require__(6395);

var ITERATOR = wellKnownSymbol('iterator');

module.exports = !fails(function () {
  // eslint-disable-next-line unicorn/relative-url-style -- required for testing
  var url = new URL('b?a=1&b=2&c=3', 'https://a');
  var params = url.searchParams;
  var params2 = new URLSearchParams('a=1&a=2&b=3');
  var result = '';
  url.pathname = 'c%20d';
  params.forEach(function (value, key) {
    params['delete']('b');
    result += key + value;
  });
  params2['delete']('a', 2);
  // `undefined` case is a Chromium 117 bug
  // https://bugs.chromium.org/p/v8/issues/detail?id=14222
  params2['delete']('b', undefined);
  return (IS_PURE && (!url.toJSON || !params2.has('a', 1) || params2.has('a', 2) || !params2.has('a', undefined) || params2.has('b')))
    || (!params.size && (IS_PURE || !DESCRIPTORS))
    || !params.sort
    || url.href !== 'https://a/c%20d?a=1&c=3'
    || params.get('c') !== '3'
    || String(new URLSearchParams('?a=1')) !== 'a=1'
    || !params[ITERATOR]
    // throws in Edge
    || new URL('https://a@b').username !== 'a'
    || new URLSearchParams(new URLSearchParams('a=b')).get('a') !== 'b'
    // not punycoded in Edge
    || new URL('https://тест').host !== 'xn--e1aybc'
    // not escaped in Chrome 62-
    || new URL('https://a#б').hash !== '#%D0%B1'
    // fails in Chrome 66-
    || result !== 'a1c3'
    // throws in Safari
    || new URL('https://x', undefined).host !== 'x';
});


/***/ },

/***/ 7040
(module, __unused_webpack_exports, __webpack_require__) {


/* eslint-disable es/no-symbol -- required for testing */
var NATIVE_SYMBOL = __webpack_require__(4495);

module.exports = NATIVE_SYMBOL &&
  !Symbol.sham &&
  typeof Symbol.iterator == 'symbol';


/***/ },

/***/ 8686
(module, __unused_webpack_exports, __webpack_require__) {


var DESCRIPTORS = __webpack_require__(3724);
var fails = __webpack_require__(9039);

// V8 ~ Chrome 36-
// https://bugs.chromium.org/p/v8/issues/detail?id=3334
module.exports = DESCRIPTORS && fails(function () {
  // eslint-disable-next-line es/no-object-defineproperty -- required for testing
  return Object.defineProperty(function () { /* empty */ }, 'prototype', {
    value: 42,
    writable: false
  }).prototype !== 42;
});


/***/ },

/***/ 2812
(module) {


var $TypeError = TypeError;

module.exports = function (passed, required) {
  if (passed < required) throw new $TypeError('Not enough arguments');
  return passed;
};


/***/ },

/***/ 8622
(module, __unused_webpack_exports, __webpack_require__) {


var globalThis = __webpack_require__(4576);
var isCallable = __webpack_require__(4901);

var WeakMap = globalThis.WeakMap;

module.exports = isCallable(WeakMap) && /native code/.test(String(WeakMap));


/***/ },

/***/ 4995
(module, __unused_webpack_exports, __webpack_require__) {


var uncurryThis = __webpack_require__(9504);

// eslint-disable-next-line es/no-weak-map -- safe
var WeakMapPrototype = WeakMap.prototype;

module.exports = {
  // eslint-disable-next-line es/no-weak-map -- safe
  WeakMap: WeakMap,
  set: uncurryThis(WeakMapPrototype.set),
  get: uncurryThis(WeakMapPrototype.get),
  has: uncurryThis(WeakMapPrototype.has),
  remove: uncurryThis(WeakMapPrototype['delete'])
};


/***/ },

/***/ 8227
(module, __unused_webpack_exports, __webpack_require__) {


var globalThis = __webpack_require__(4576);
var shared = __webpack_require__(5745);
var hasOwn = __webpack_require__(9297);
var uid = __webpack_require__(3392);
var NATIVE_SYMBOL = __webpack_require__(4495);
var USE_SYMBOL_AS_UID = __webpack_require__(7040);

var Symbol = globalThis.Symbol;
var WellKnownSymbolsStore = shared('wks');
var createWellKnownSymbol = USE_SYMBOL_AS_UID ? Symbol['for'] || Symbol : Symbol && Symbol.withoutSetter || uid;

module.exports = function (name) {
  if (!hasOwn(WellKnownSymbolsStore, name)) {
    WellKnownSymbolsStore[name] = NATIVE_SYMBOL && hasOwn(Symbol, name)
      ? Symbol[name]
      : createWellKnownSymbol('Symbol.' + name);
  } return WellKnownSymbolsStore[name];
};


/***/ },

/***/ 7452
(module) {


// a string of all valid unicode whitespaces
module.exports = '\u0009\u000A\u000B\u000C\u000D\u0020\u00A0\u1680\u2000\u2001\u2002' +
  '\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200A\u202F\u205F\u3000\u2028\u2029\uFEFF';


/***/ },

/***/ 4423
(__unused_webpack_module, __unused_webpack_exports, __webpack_require__) {


var $ = __webpack_require__(6518);
var $includes = (__webpack_require__(9617).includes);
var fails = __webpack_require__(9039);
var addToUnscopables = __webpack_require__(6469);

// FF99+ bug
var BROKEN_ON_SPARSE = fails(function () {
  // eslint-disable-next-line es/no-array-prototype-includes -- detection
  return !Array(1).includes();
});

// Safari 26.4- bug
var BROKEN_ON_SPARSE_WITH_FROM_INDEX = fails(function () {
  // eslint-disable-next-line no-sparse-arrays, es/no-array-prototype-includes -- detection
  return [, 1].includes(undefined, 1);
});

// `Array.prototype.includes` method
// https://tc39.es/ecma262/#sec-array.prototype.includes
$({ target: 'Array', proto: true, forced: BROKEN_ON_SPARSE || BROKEN_ON_SPARSE_WITH_FROM_INDEX }, {
  includes: function includes(el /* , fromIndex = 0 */) {
    return $includes(this, el, arguments.length > 1 ? arguments[1] : undefined);
  }
});

// https://tc39.es/ecma262/#sec-array.prototype-@@unscopables
addToUnscopables('includes');


/***/ },

/***/ 4114
(__unused_webpack_module, __unused_webpack_exports, __webpack_require__) {


var $ = __webpack_require__(6518);
var toObject = __webpack_require__(8981);
var lengthOfArrayLike = __webpack_require__(6198);
var setArrayLength = __webpack_require__(4527);
var doesNotExceedSafeInteger = __webpack_require__(6837);
var fails = __webpack_require__(9039);

var INCORRECT_TO_LENGTH = fails(function () {
  return [].push.call({ length: 0x100000000 }, 1) !== 4294967297;
});

// V8 <= 121 and Safari <= 15.4; FF < 23 throws InternalError
// https://bugs.chromium.org/p/v8/issues/detail?id=12681
var properErrorOnNonWritableLength = function () {
  try {
    // eslint-disable-next-line es/no-object-defineproperty -- safe
    Object.defineProperty([], 'length', { writable: false }).push();
  } catch (error) {
    return error instanceof TypeError;
  }
};

var FORCED = INCORRECT_TO_LENGTH || !properErrorOnNonWritableLength();

// `Array.prototype.push` method
// https://tc39.es/ecma262/#sec-array.prototype.push
$({ target: 'Array', proto: true, arity: 1, forced: FORCED }, {
  // eslint-disable-next-line no-unused-vars -- required for `.length`
  push: function push(item) {
    var O = toObject(this);
    var len = lengthOfArrayLike(O);
    var argCount = arguments.length;
    doesNotExceedSafeInteger(len + argCount);
    for (var i = 0; i < argCount; i++) {
      O[len] = arguments[i];
      len++;
    }
    setArrayLength(O, len);
    return len;
  }
});


/***/ },

/***/ 8111
(__unused_webpack_module, __unused_webpack_exports, __webpack_require__) {


var $ = __webpack_require__(6518);
var globalThis = __webpack_require__(4576);
var anInstance = __webpack_require__(679);
var anObject = __webpack_require__(8551);
var isCallable = __webpack_require__(4901);
var getPrototypeOf = __webpack_require__(2787);
var defineBuiltInAccessor = __webpack_require__(2106);
var createProperty = __webpack_require__(4659);
var fails = __webpack_require__(9039);
var hasOwn = __webpack_require__(9297);
var wellKnownSymbol = __webpack_require__(8227);
var IteratorPrototype = (__webpack_require__(7657).IteratorPrototype);
var DESCRIPTORS = __webpack_require__(3724);
var IS_PURE = __webpack_require__(6395);

var CONSTRUCTOR = 'constructor';
var ITERATOR = 'Iterator';
var TO_STRING_TAG = wellKnownSymbol('toStringTag');

var $TypeError = TypeError;
var NativeIterator = globalThis[ITERATOR];

// FF56- have non-standard global helper `Iterator`
var FORCED = IS_PURE
  || !isCallable(NativeIterator)
  || NativeIterator.prototype !== IteratorPrototype
  // FF44- non-standard `Iterator` passes previous tests
  || !fails(function () { NativeIterator({}); });

var IteratorConstructor = function Iterator() {
  anInstance(this, IteratorPrototype);
  if (getPrototypeOf(this) === IteratorPrototype) throw new $TypeError('Abstract class Iterator not directly constructable');
};

var defineIteratorPrototypeAccessor = function (key, value) {
  if (DESCRIPTORS) {
    defineBuiltInAccessor(IteratorPrototype, key, {
      configurable: true,
      get: function () {
        return value;
      },
      set: function (replacement) {
        anObject(this);
        if (this === IteratorPrototype) throw new $TypeError("You can't redefine this property");
        if (hasOwn(this, key)) this[key] = replacement;
        else createProperty(this, key, replacement);
      }
    });
  } else IteratorPrototype[key] = value;
};

if (!hasOwn(IteratorPrototype, TO_STRING_TAG)) defineIteratorPrototypeAccessor(TO_STRING_TAG, ITERATOR);

if (FORCED || !hasOwn(IteratorPrototype, CONSTRUCTOR) || IteratorPrototype[CONSTRUCTOR] === Object) {
  defineIteratorPrototypeAccessor(CONSTRUCTOR, IteratorConstructor);
}

IteratorConstructor.prototype = IteratorPrototype;

// `Iterator` constructor
// https://tc39.es/ecma262/#sec-iterator
$({ global: true, constructor: true, forced: FORCED }, {
  Iterator: IteratorConstructor
});


/***/ },

/***/ 1148
(__unused_webpack_module, __unused_webpack_exports, __webpack_require__) {


var $ = __webpack_require__(6518);
var call = __webpack_require__(9565);
var iterate = __webpack_require__(2652);
var aCallable = __webpack_require__(9306);
var anObject = __webpack_require__(8551);
var getIteratorDirect = __webpack_require__(1767);
var iteratorClose = __webpack_require__(9539);
var iteratorHelperWithoutClosingOnEarlyError = __webpack_require__(4549);

var everyWithoutClosingOnEarlyError = iteratorHelperWithoutClosingOnEarlyError('every', TypeError);

// `Iterator.prototype.every` method
// https://tc39.es/ecma262/#sec-iterator.prototype.every
$({ target: 'Iterator', proto: true, real: true, forced: everyWithoutClosingOnEarlyError }, {
  every: function every(predicate) {
    anObject(this);
    try {
      aCallable(predicate);
    } catch (error) {
      iteratorClose(this, 'throw', error);
    }

    if (everyWithoutClosingOnEarlyError) return call(everyWithoutClosingOnEarlyError, this, predicate);

    var record = getIteratorDirect(this);
    var counter = 0;
    return !iterate(record, function (value, stop) {
      if (!predicate(value, counter++)) return stop();
    }, { IS_RECORD: true, INTERRUPTED: true }).stopped;
  }
});


/***/ },

/***/ 2489
(__unused_webpack_module, __unused_webpack_exports, __webpack_require__) {


var $ = __webpack_require__(6518);
var call = __webpack_require__(9565);
var aCallable = __webpack_require__(9306);
var anObject = __webpack_require__(8551);
var getIteratorDirect = __webpack_require__(1767);
var createIteratorProxy = __webpack_require__(9462);
var callWithSafeIterationClosing = __webpack_require__(6319);
var IS_PURE = __webpack_require__(6395);
var iteratorClose = __webpack_require__(9539);
var iteratorHelperThrowsOnInvalidIterator = __webpack_require__(684);
var iteratorHelperWithoutClosingOnEarlyError = __webpack_require__(4549);

var FILTER_WITHOUT_THROWING_ON_INVALID_ITERATOR = !IS_PURE && !iteratorHelperThrowsOnInvalidIterator('filter', function () { /* empty */ });
var filterWithoutClosingOnEarlyError = !IS_PURE && !FILTER_WITHOUT_THROWING_ON_INVALID_ITERATOR
  && iteratorHelperWithoutClosingOnEarlyError('filter', TypeError);

var FORCED = IS_PURE || FILTER_WITHOUT_THROWING_ON_INVALID_ITERATOR || filterWithoutClosingOnEarlyError;

var IteratorProxy = createIteratorProxy(function () {
  var iterator = this.iterator;
  var predicate = this.predicate;
  var next = this.next;
  var result, done, value;
  while (true) {
    result = anObject(call(next, iterator));
    done = this.done = !!result.done;
    if (done) return;
    value = result.value;
    if (callWithSafeIterationClosing(iterator, predicate, [value, this.counter++], true)) return value;
  }
});

// `Iterator.prototype.filter` method
// https://tc39.es/ecma262/#sec-iterator.prototype.filter
$({ target: 'Iterator', proto: true, real: true, forced: FORCED }, {
  filter: function filter(predicate) {
    anObject(this);
    try {
      aCallable(predicate);
    } catch (error) {
      iteratorClose(this, 'throw', error);
    }

    if (filterWithoutClosingOnEarlyError) return call(filterWithoutClosingOnEarlyError, this, predicate);

    return new IteratorProxy(getIteratorDirect(this), {
      predicate: predicate
    });
  }
});


/***/ },

/***/ 116
(__unused_webpack_module, __unused_webpack_exports, __webpack_require__) {


var $ = __webpack_require__(6518);
var call = __webpack_require__(9565);
var iterate = __webpack_require__(2652);
var aCallable = __webpack_require__(9306);
var anObject = __webpack_require__(8551);
var getIteratorDirect = __webpack_require__(1767);
var iteratorClose = __webpack_require__(9539);
var iteratorHelperWithoutClosingOnEarlyError = __webpack_require__(4549);

var findWithoutClosingOnEarlyError = iteratorHelperWithoutClosingOnEarlyError('find', TypeError);

// `Iterator.prototype.find` method
// https://tc39.es/ecma262/#sec-iterator.prototype.find
$({ target: 'Iterator', proto: true, real: true, forced: findWithoutClosingOnEarlyError }, {
  find: function find(predicate) {
    anObject(this);
    try {
      aCallable(predicate);
    } catch (error) {
      iteratorClose(this, 'throw', error);
    }

    if (findWithoutClosingOnEarlyError) return call(findWithoutClosingOnEarlyError, this, predicate);

    var record = getIteratorDirect(this);
    var counter = 0;
    return iterate(record, function (value, stop) {
      if (predicate(value, counter++)) return stop(value);
    }, { IS_RECORD: true, INTERRUPTED: true }).result;
  }
});


/***/ },

/***/ 7588
(__unused_webpack_module, __unused_webpack_exports, __webpack_require__) {


var $ = __webpack_require__(6518);
var call = __webpack_require__(9565);
var iterate = __webpack_require__(2652);
var aCallable = __webpack_require__(9306);
var anObject = __webpack_require__(8551);
var getIteratorDirect = __webpack_require__(1767);
var iteratorClose = __webpack_require__(9539);
var iteratorHelperWithoutClosingOnEarlyError = __webpack_require__(4549);

var forEachWithoutClosingOnEarlyError = iteratorHelperWithoutClosingOnEarlyError('forEach', TypeError);

// `Iterator.prototype.forEach` method
// https://tc39.es/ecma262/#sec-iterator.prototype.foreach
$({ target: 'Iterator', proto: true, real: true, forced: forEachWithoutClosingOnEarlyError }, {
  forEach: function forEach(fn) {
    anObject(this);
    try {
      aCallable(fn);
    } catch (error) {
      iteratorClose(this, 'throw', error);
    }

    if (forEachWithoutClosingOnEarlyError) return call(forEachWithoutClosingOnEarlyError, this, fn);

    var record = getIteratorDirect(this);
    var counter = 0;
    iterate(record, function (value) {
      fn(value, counter++);
    }, { IS_RECORD: true });
  }
});


/***/ },

/***/ 1701
(__unused_webpack_module, __unused_webpack_exports, __webpack_require__) {


var $ = __webpack_require__(6518);
var call = __webpack_require__(9565);
var aCallable = __webpack_require__(9306);
var anObject = __webpack_require__(8551);
var getIteratorDirect = __webpack_require__(1767);
var createIteratorProxy = __webpack_require__(9462);
var callWithSafeIterationClosing = __webpack_require__(6319);
var iteratorClose = __webpack_require__(9539);
var iteratorHelperThrowsOnInvalidIterator = __webpack_require__(684);
var iteratorHelperWithoutClosingOnEarlyError = __webpack_require__(4549);
var IS_PURE = __webpack_require__(6395);

var MAP_WITHOUT_THROWING_ON_INVALID_ITERATOR = !IS_PURE && !iteratorHelperThrowsOnInvalidIterator('map', function () { /* empty */ });
var mapWithoutClosingOnEarlyError = !IS_PURE && !MAP_WITHOUT_THROWING_ON_INVALID_ITERATOR
  && iteratorHelperWithoutClosingOnEarlyError('map', TypeError);

var FORCED = IS_PURE || MAP_WITHOUT_THROWING_ON_INVALID_ITERATOR || mapWithoutClosingOnEarlyError;

var IteratorProxy = createIteratorProxy(function () {
  var iterator = this.iterator;
  var result = anObject(call(this.next, iterator));
  var done = this.done = !!result.done;
  if (!done) return callWithSafeIterationClosing(iterator, this.mapper, [result.value, this.counter++], true);
});

// `Iterator.prototype.map` method
// https://tc39.es/ecma262/#sec-iterator.prototype.map
$({ target: 'Iterator', proto: true, real: true, forced: FORCED }, {
  map: function map(mapper) {
    anObject(this);
    try {
      aCallable(mapper);
    } catch (error) {
      iteratorClose(this, 'throw', error);
    }

    if (mapWithoutClosingOnEarlyError) return call(mapWithoutClosingOnEarlyError, this, mapper);

    return new IteratorProxy(getIteratorDirect(this), {
      mapper: mapper
    });
  }
});


/***/ },

/***/ 3579
(__unused_webpack_module, __unused_webpack_exports, __webpack_require__) {


var $ = __webpack_require__(6518);
var call = __webpack_require__(9565);
var iterate = __webpack_require__(2652);
var aCallable = __webpack_require__(9306);
var anObject = __webpack_require__(8551);
var getIteratorDirect = __webpack_require__(1767);
var iteratorClose = __webpack_require__(9539);
var iteratorHelperWithoutClosingOnEarlyError = __webpack_require__(4549);

var someWithoutClosingOnEarlyError = iteratorHelperWithoutClosingOnEarlyError('some', TypeError);

// `Iterator.prototype.some` method
// https://tc39.es/ecma262/#sec-iterator.prototype.some
$({ target: 'Iterator', proto: true, real: true, forced: someWithoutClosingOnEarlyError }, {
  some: function some(predicate) {
    anObject(this);
    try {
      aCallable(predicate);
    } catch (error) {
      iteratorClose(this, 'throw', error);
    }

    if (someWithoutClosingOnEarlyError) return call(someWithoutClosingOnEarlyError, this, predicate);

    var record = getIteratorDirect(this);
    var counter = 0;
    return iterate(record, function (value, stop) {
      if (predicate(value, counter++)) return stop();
    }, { IS_RECORD: true, INTERRUPTED: true }).stopped;
  }
});


/***/ },

/***/ 9112
(__unused_webpack_module, __unused_webpack_exports, __webpack_require__) {


var $ = __webpack_require__(6518);
var DESCRIPTORS = __webpack_require__(3724);
var globalThis = __webpack_require__(4576);
var getBuiltIn = __webpack_require__(7751);
var uncurryThis = __webpack_require__(9504);
var call = __webpack_require__(9565);
var isCallable = __webpack_require__(4901);
var isObject = __webpack_require__(34);
var isArray = __webpack_require__(4376);
var hasOwn = __webpack_require__(9297);
var toString = __webpack_require__(655);
var lengthOfArrayLike = __webpack_require__(6198);
var createProperty = __webpack_require__(4659);
var fails = __webpack_require__(9039);
var parseJSONString = __webpack_require__(8235);
var NATIVE_SYMBOL = __webpack_require__(4495);

var JSON = globalThis.JSON;
var Number = globalThis.Number;
var SyntaxError = globalThis.SyntaxError;
var nativeParse = JSON && JSON.parse;
var enumerableOwnProperties = getBuiltIn('Object', 'keys');
// eslint-disable-next-line es/no-object-getownpropertydescriptor -- safe
var getOwnPropertyDescriptor = Object.getOwnPropertyDescriptor;
var at = uncurryThis(''.charAt);
var slice = uncurryThis(''.slice);
var exec = uncurryThis(/./.exec);
var push = uncurryThis([].push);

var IS_DIGIT = /^\d$/;
var IS_NON_ZERO_DIGIT = /^[1-9]$/;
var IS_NUMBER_START = /^[\d-]$/;
var IS_WHITESPACE = /^[\t\n\r ]$/;

var PRIMITIVE = 0;
var OBJECT = 1;

var $parse = function (source, reviver) {
  source = toString(source);
  var context = new Context(source, 0);
  var root = context.parse();
  var value = root.value;
  var endIndex = context.skip(IS_WHITESPACE, root.end);
  if (endIndex < source.length) {
    throw new SyntaxError('Unexpected extra character: "' + at(source, endIndex) + '" after the parsed data at: ' + endIndex);
  }
  return isCallable(reviver) ? internalize({ '': value }, '', reviver, root) : value;
};

var internalize = function (holder, name, reviver, node) {
  var val = holder[name];
  var unmodified = node && val === node.value;
  var context = unmodified && typeof node.source == 'string' ? { source: node.source } : {};
  var elementRecordsLen, keys, len, i, P;
  if (isObject(val)) {
    var nodeIsArray = isArray(val);
    var nodes = unmodified ? node.nodes : nodeIsArray ? [] : {};
    if (nodeIsArray) {
      elementRecordsLen = nodes.length;
      len = lengthOfArrayLike(val);
      for (i = 0; i < len; i++) {
        internalizeProperty(val, i, internalize(val, '' + i, reviver, i < elementRecordsLen ? nodes[i] : undefined));
      }
    } else {
      keys = enumerableOwnProperties(val);
      len = lengthOfArrayLike(keys);
      for (i = 0; i < len; i++) {
        P = keys[i];
        internalizeProperty(val, P, internalize(val, P, reviver, hasOwn(nodes, P) ? nodes[P] : undefined));
      }
    }
  }
  return call(reviver, holder, name, val, context);
};

var internalizeProperty = function (object, key, value) {
  if (DESCRIPTORS) {
    var descriptor = getOwnPropertyDescriptor(object, key);
    if (descriptor && !descriptor.configurable) return;
  }
  if (value === undefined) delete object[key];
  else createProperty(object, key, value);
};

var Node = function (value, end, source, nodes) {
  this.value = value;
  this.end = end;
  this.source = source;
  this.nodes = nodes;
};

var Context = function (source, index) {
  this.source = source;
  this.index = index;
};

// https://www.json.org/json-en.html
Context.prototype = {
  fork: function (nextIndex) {
    return new Context(this.source, nextIndex);
  },
  parse: function () {
    var source = this.source;
    var i = this.skip(IS_WHITESPACE, this.index);
    var fork = this.fork(i);
    var chr = at(source, i);
    if (exec(IS_NUMBER_START, chr)) return fork.number();
    switch (chr) {
      case '{':
        return fork.object();
      case '[':
        return fork.array();
      case '"':
        return fork.string();
      case 't':
        return fork.keyword(true);
      case 'f':
        return fork.keyword(false);
      case 'n':
        return fork.keyword(null);
    } throw new SyntaxError('Unexpected character: "' + chr + '" at: ' + i);
  },
  node: function (type, value, start, end, nodes) {
    return new Node(value, end, type ? null : slice(this.source, start, end), nodes);
  },
  object: function () {
    var source = this.source;
    var i = this.index + 1;
    var expectKeypair = false;
    var object = {};
    var nodes = {};
    var closed = false;
    while (i < source.length) {
      i = this.until(['"', '}'], i);
      if (at(source, i) === '}' && !expectKeypair) {
        i++;
        closed = true;
        break;
      }
      // Parsing the key
      var result = this.fork(i).string();
      var key = result.value;
      i = result.end;
      i = this.until([':'], i) + 1;
      // Parsing value
      i = this.skip(IS_WHITESPACE, i);
      result = this.fork(i).parse();
      createProperty(nodes, key, result);
      createProperty(object, key, result.value);
      i = this.until([',', '}'], result.end);
      var chr = at(source, i);
      if (chr === ',') {
        expectKeypair = true;
        i++;
      } else if (chr === '}') {
        i++;
        closed = true;
        break;
      }
    }
    if (!closed) throw new SyntaxError('Unterminated object at: ' + i);
    return this.node(OBJECT, object, this.index, i, nodes);
  },
  array: function () {
    var source = this.source;
    var i = this.index + 1;
    var expectElement = false;
    var array = [];
    var nodes = [];
    var closed = false;
    while (i < source.length) {
      i = this.skip(IS_WHITESPACE, i);
      if (at(source, i) === ']' && !expectElement) {
        i++;
        closed = true;
        break;
      }
      var result = this.fork(i).parse();
      push(nodes, result);
      push(array, result.value);
      i = this.until([',', ']'], result.end);
      if (at(source, i) === ',') {
        expectElement = true;
        i++;
      } else if (at(source, i) === ']') {
        i++;
        closed = true;
        break;
      }
    }
    if (!closed) throw new SyntaxError('Unterminated array at: ' + i);
    return this.node(OBJECT, array, this.index, i, nodes);
  },
  string: function () {
    var index = this.index;
    var parsed = parseJSONString(this.source, this.index + 1);
    return this.node(PRIMITIVE, parsed.value, index, parsed.end);
  },
  number: function () {
    var source = this.source;
    var startIndex = this.index;
    var i = startIndex;
    if (at(source, i) === '-') i++;
    if (at(source, i) === '0') i++;
    else if (exec(IS_NON_ZERO_DIGIT, at(source, i))) i = this.skip(IS_DIGIT, i + 1);
    else throw new SyntaxError('Failed to parse number at: ' + i);
    if (at(source, i) === '.') {
      var fractionStartIndex = i + 1;
      i = this.skip(IS_DIGIT, fractionStartIndex);
      if (fractionStartIndex === i) throw new SyntaxError("Failed to parse number's fraction at: " + i);
    }
    if (at(source, i) === 'e' || at(source, i) === 'E') {
      i++;
      if (at(source, i) === '+' || at(source, i) === '-') i++;
      var exponentStartIndex = i;
      i = this.skip(IS_DIGIT, i);
      if (exponentStartIndex === i) throw new SyntaxError("Failed to parse number's exponent value at: " + i);
    }
    return this.node(PRIMITIVE, Number(slice(source, startIndex, i)), startIndex, i);
  },
  keyword: function (value) {
    var keyword = '' + value;
    var index = this.index;
    var endIndex = index + keyword.length;
    if (slice(this.source, index, endIndex) !== keyword) throw new SyntaxError('Failed to parse value at: ' + index);
    return this.node(PRIMITIVE, value, index, endIndex);
  },
  skip: function (regex, i) {
    var source = this.source;
    for (; i < source.length; i++) if (!exec(regex, at(source, i))) break;
    return i;
  },
  until: function (array, i) {
    i = this.skip(IS_WHITESPACE, i);
    var chr = at(this.source, i);
    for (var j = 0; j < array.length; j++) if (array[j] === chr) return i;
    throw new SyntaxError('Unexpected character: "' + chr + '" at: ' + i);
  }
};

var NO_SOURCE_SUPPORT = fails(function () {
  var unsafeInt = '9007199254740993';
  var source;
  nativeParse(unsafeInt, function (key, value, context) {
    source = context.source;
  });
  return source !== unsafeInt;
});

var PROPER_BASE_PARSE = NATIVE_SYMBOL && !fails(function () {
  // Safari 9 bug
  return 1 / nativeParse('-0 \t') !== -Infinity;
});

// `JSON.parse` method
// https://tc39.es/ecma262/#sec-json.parse
// https://github.com/tc39/proposal-json-parse-with-source
$({ target: 'JSON', stat: true, forced: NO_SOURCE_SUPPORT }, {
  parse: function parse(text, reviver) {
    return PROPER_BASE_PARSE && !isCallable(reviver) ? nativeParse(text) : $parse(text, reviver);
  }
});


/***/ },

/***/ 3110
(__unused_webpack_module, __unused_webpack_exports, __webpack_require__) {


var $ = __webpack_require__(6518);
var getBuiltIn = __webpack_require__(7751);
var call = __webpack_require__(9565);
var uncurryThis = __webpack_require__(9504);
var fails = __webpack_require__(9039);
var isArray = __webpack_require__(4376);
var isCallable = __webpack_require__(4901);
var isObject = __webpack_require__(34);
var create = __webpack_require__(2360);
var isRawJSON = __webpack_require__(5810);
var isSymbol = __webpack_require__(757);
var classof = __webpack_require__(2195);
var thisNumberValue = __webpack_require__(1240);
var includes = (__webpack_require__(9617).includes);
var hasOwn = __webpack_require__(9297);
var toString = __webpack_require__(655);
var parseJSONString = __webpack_require__(8235);
var uid = __webpack_require__(3392);
var NATIVE_SYMBOL = __webpack_require__(4495);
var NATIVE_RAW_JSON = __webpack_require__(7819);

var $String = String;
var $TypeError = TypeError;
var $stringify = getBuiltIn('JSON', 'stringify');
var $BigInt = getBuiltIn('BigInt');
var stringValueOf = uncurryThis(''.valueOf);
var booleanValueOf = uncurryThis(true.valueOf);
var bigIntValueOf = $BigInt && uncurryThis($BigInt.prototype.valueOf);
var exec = uncurryThis(/./.exec);
var charAt = uncurryThis(''.charAt);
var charCodeAt = uncurryThis(''.charCodeAt);
var replace = uncurryThis(''.replace);
var slice = uncurryThis(''.slice);
var push = uncurryThis([].push);
var pop = uncurryThis([].pop);
var numberToString = uncurryThis(1.1.toString);

var surrogates = /[\uD800-\uDFFF]/g;
var leadingSurrogates = /^[\uD800-\uDBFF]$/;
var trailingSurrogates = /^[\uDC00-\uDFFF]$/;
var digits = /^\d+$/;

// a placeholder of a raw JSON value
var RAW_MARK = uid();
// a prefix of keys of a reordered object, see `createOrderedObject`
var KEY_MARK = uid();
// the last key of a reordered object, marks the end of its serialization
var END_MARK = uid();
var RAW_MARK_LENGTH = RAW_MARK.length;
var KEY_MARK_LENGTH = KEY_MARK.length;

var WRONG_SYMBOLS_CONVERSION = !NATIVE_SYMBOL || fails(function () {
  var symbol = getBuiltIn('Symbol')('stringify detection');
  // MS Edge converts symbol values to JSON as {}
  return $stringify([symbol]) !== '[null]'
    // WebKit converts symbol values to JSON as null
    || $stringify({ a: symbol }) !== '{}'
    // V8 throws on boxed symbols
    || $stringify(Object(symbol)) !== '{}';
});

// https://github.com/tc39/proposal-well-formed-stringify
var ILL_FORMED_UNICODE = fails(function () {
  return $stringify('\uDF06\uD834') !== '"\\udf06\\ud834"'
    || $stringify('\uDEAD') !== '"\\udead"';
});

var isRawJSONValue = NATIVE_RAW_JSON ? getBuiltIn('JSON', 'isRawJSON') : isRawJSON;

var stringifyWithProperSymbolsConversion = WRONG_SYMBOLS_CONVERSION ? function (it, replacer, space) {
  return $stringify(it, function (key, value) {
    var replaced = call(replacer, this, key, value);
    if (!isSymbol(replaced)) return replaced;
  }, space);
} : $stringify;

var fixIllFormedJSON = function (match, offset, string) {
  var prev = charAt(string, offset - 1);
  var next = charAt(string, offset + 1);
  if (
    (exec(leadingSurrogates, match) && !exec(trailingSurrogates, next)) ||
    (exec(trailingSurrogates, match) && !exec(leadingSurrogates, prev))
  ) {
    return '\\u' + numberToString(charCodeAt(match, 0), 16);
  } return match;
};

// `PropertyList` of `JSON.stringify`
// https://tc39.es/ecma262/#sec-json.stringify
var getPropertyList = function (replacer) {
  if (!isArray(replacer)) return;
  var rawLength = replacer.length;
  var propertyList = [];
  // a null prototype object is used as a set of already added keys to keep the deduplication linear
  var addedKeys = create(null);
  for (var i = 0; i < rawLength; i++) {
    var element = replacer[i];
    var key;
    if (typeof element == 'string') key = element;
    else if (typeof element == 'number' || classof(element) === 'Number' || classof(element) === 'String') key = toString(element);
    else continue;
    if (!hasOwn(addedKeys, key)) {
      addedKeys[key] = true;
      push(propertyList, key);
    }
  }
  return propertyList;
};

// values with such an internal slot are unwrapped by `SerializeJSONProperty` instead of being serialized as objects
var hasInternalSlot = function (valueOf, it) {
  try {
    valueOf(it);
    return true;
  } catch (error) {
    return false;
  }
};

// the slot check is expensive, so it's performed only for the kind reported by the value itself -
// a value lying about its kind via `Symbol.toStringTag` is serialized as an ordinary object
var isBoxedPrimitive = function (it) {
  var kind = classof(it);
  return (kind === 'Number' && hasInternalSlot(thisNumberValue, it))
    || (kind === 'String' && hasInternalSlot(stringValueOf, it))
    || (kind === 'Boolean' && hasInternalSlot(booleanValueOf, it))
    || (!!bigIntValueOf && kind === 'BigInt' && hasInternalSlot(bigIntValueOf, it));
};

// only objects serialized by `SerializeJSONObject` are affected by the property list
var isSerializedAsObject = function (it) {
  if (!isObject(it) || isCallable(it) || isArray(it)) return false;
  try {
    return !isBoxedPrimitive(it);
  // `classof` reads `Symbol.toStringTag`, so a proxy could throw - it has no internal slots anyway
  } catch (error) {
    return true;
  }
};

// the engine unwraps it in the same order as it would read the original property,
// so the property is read lazily and `toJSON` is called once and with the original key
var createElementHolder = function (holder, key) {
  return {
    toJSON: function () {
      var element = holder[key];
      if (isObject(element) || typeof element == 'bigint') {
        var elementToJSON = element.toJSON;
        if (isCallable(elementToJSON)) element = call(elementToJSON, element, key);
      } return element;
    }
  };
};

// own keys of objects are sorted - integer-like keys are moved to the beginning,
// so such keys should be marked and restored in the serialized string
var getKeyPrefix = function (propertyList) {
  for (var i = 0, length = propertyList.length; i < length; i++) {
    if (exec(digits, propertyList[i])) return KEY_MARK;
  } return '';
};

// `SerializeJSONObject` iterates the property list, so the value is replaced with an object with keys in this order
var createOrderedObject = function (value, propertyList, keyPrefix) {
  // keys are not marked if the property list has no integer-like keys, so `Object.prototype`
  // with a setter, a non-writable property or `__proto__` should not intercept the assignment
  var ordered = create(null);
  for (var i = 0, length = propertyList.length; i < length; i++) {
    var key = propertyList[i];
    ordered[keyPrefix + key] = createElementHolder(value, key);
  }
  ordered[END_MARK] = null;
  return ordered;
};

// `JSON.stringify` method
// https://tc39.es/ecma262/#sec-json.stringify
// https://github.com/tc39/proposal-json-parse-with-source
if ($stringify) $({ target: 'JSON', stat: true, arity: 3, forced: WRONG_SYMBOLS_CONVERSION || ILL_FORMED_UNICODE || !NATIVE_RAW_JSON }, {
  stringify: function stringify(text, replacer, space) {
    var replacerFunction = isCallable(replacer) ? replacer : undefined;
    var propertyList = replacerFunction ? undefined : getPropertyList(replacer);
    var keyPrefix = propertyList && getKeyPrefix(propertyList);
    var rawStrings = [];
    var openObjects = [];
    var parentOrdered = [];
    var currentOrdered;
    var marked = false;
    var root = true;

    var json = stringifyWithProperSymbolsConversion(text, function (key, value) {
      // some old implementations (like WebKit) could pass numbers as keys
      key = $String(key);

      if (propertyList) {
        if (key === END_MARK) {
          pop(openObjects);
          currentOrdered = pop(parentOrdered);
          return;
        }
        if (root) root = false;
        // the innermost reordered object already contains only keys of the property list and arrays are not
        // affected by it, the rest of objects (like objects with a fake `Symbol.toStringTag`) are filtered here
        else if (this !== currentOrdered && !isArray(this) && !includes(propertyList, key)) return;
      } else if (replacerFunction) value = call(replacerFunction, this, key, value);

      if (isRawJSONValue(value)) {
        if (NATIVE_RAW_JSON) return value;
        marked = true;
        return RAW_MARK + (push(rawStrings, value.rawJSON) - 1);
      }

      if (propertyList && isSerializedAsObject(value)) {
        // reordered objects are new each time, so cycles should be detected before the engine does it
        if (includes(openObjects, value)) throw new $TypeError('Converting circular structure to JSON');
        var ordered = createOrderedObject(value, propertyList, keyPrefix);
        push(openObjects, value);
        push(parentOrdered, currentOrdered);
        currentOrdered = ordered;
        if (keyPrefix) marked = true;
        return ordered;
      }

      return value;
    }, space);

    if (typeof json != 'string') return json;

    if (ILL_FORMED_UNICODE) json = replace(json, surrogates, fixIllFormedJSON);

    if (!marked) return json;

    var result = '';
    var length = json.length;

    for (var i = 0; i < length; i++) {
      var chr = charAt(json, i);
      if (chr === '"') {
        var end = parseJSONString(json, ++i).end - 1;
        var string = slice(json, i, end);
        if (slice(string, 0, RAW_MARK_LENGTH) === RAW_MARK) result += rawStrings[slice(string, RAW_MARK_LENGTH)];
        else if (slice(string, 0, KEY_MARK_LENGTH) === KEY_MARK) result += '"' + slice(string, KEY_MARK_LENGTH) + '"';
        else result += '"' + string + '"';
        i = end;
      } else result += chr;
    }

    return result;
  }
});


/***/ },

/***/ 2731
(__unused_webpack_module, __unused_webpack_exports, __webpack_require__) {


var $ = __webpack_require__(6518);
var aCallable = __webpack_require__(9306);
var MapHelpers = __webpack_require__(2248);
var IS_PURE = __webpack_require__(6395);

var get = MapHelpers.get;
var has = MapHelpers.has;
var set = MapHelpers.set;

// `Map.prototype.getOrInsertComputed` method
// https://tc39.es/ecma262/#sec-map.prototype.getorinsertcomputed
$({ target: 'Map', proto: true, real: true, forced: IS_PURE }, {
  getOrInsertComputed: function getOrInsertComputed(key, callbackfn) {
    var hasKey = has(this, key);
    aCallable(callbackfn);
    if (hasKey) return get(this, key);
    // CanonicalizeKeyedCollectionKey
    if (key === 0 && 1 / key === -Infinity) key = 0;
    var value = callbackfn(key);
    set(this, key, value);
    return value;
  }
});


/***/ },

/***/ 5367
(__unused_webpack_module, __unused_webpack_exports, __webpack_require__) {


var $ = __webpack_require__(6518);
var MapHelpers = __webpack_require__(2248);
var IS_PURE = __webpack_require__(6395);

var get = MapHelpers.get;
var has = MapHelpers.has;
var set = MapHelpers.set;

// `Map.prototype.getOrInsert` method
// https://tc39.es/ecma262/#sec-map.prototype.getorinsert
$({ target: 'Map', proto: true, real: true, forced: IS_PURE }, {
  getOrInsert: function getOrInsert(key, value) {
    if (has(this, key)) return get(this, key);
    set(this, key, value);
    return value;
  }
});


/***/ },

/***/ 6069
(__unused_webpack_module, __unused_webpack_exports, __webpack_require__) {


var $ = __webpack_require__(6518);
var uncurryThis = __webpack_require__(9504);
var aString = __webpack_require__(3463);
var hasOwn = __webpack_require__(9297);
var padStart = (__webpack_require__(533).start);
var WHITESPACES = __webpack_require__(7452);

var $Array = Array;
var $escape = RegExp.escape;
var charAt = uncurryThis(''.charAt);
var charCodeAt = uncurryThis(''.charCodeAt);
var numberToString = uncurryThis(1.1.toString);
var join = uncurryThis([].join);
var FIRST_DIGIT_OR_ASCII = /^[0-9a-z]/i;
var SYNTAX_SOLIDUS = /^[$()*+./?[\\\]^{|}]/;
var OTHER_PUNCTUATORS_AND_WHITESPACES = RegExp('^[!"#%&\',\\-:;<=>@`~' + WHITESPACES + ']');
var exec = uncurryThis(FIRST_DIGIT_OR_ASCII.exec);

var ControlEscape = {
  '\u0009': 't',
  '\u000A': 'n',
  '\u000B': 'v',
  '\u000C': 'f',
  '\u000D': 'r'
};

var escapeChar = function (chr) {
  var hex = numberToString(charCodeAt(chr, 0), 16);
  return hex.length < 3 ? '\\x' + padStart(hex, 2, '0') : '\\u' + padStart(hex, 4, '0');
};

// Avoiding the use of polyfills of the previous iteration of this proposal
var FORCED = !$escape || $escape('ab') !== '\\x61b';

// `RegExp.escape` method
// https://tc39.es/ecma262/#sec-regexp.escape
$({ target: 'RegExp', stat: true, forced: FORCED }, {
  escape: function escape(S) {
    aString(S);
    var length = S.length;
    var result = $Array(length);

    for (var i = 0; i < length; i++) {
      var chr = charAt(S, i);
      if (i === 0 && exec(FIRST_DIGIT_OR_ASCII, chr)) {
        result[i] = escapeChar(chr);
      } else if (hasOwn(ControlEscape, chr)) {
        result[i] = '\\' + ControlEscape[chr];
      } else if (exec(SYNTAX_SOLIDUS, chr)) {
        result[i] = '\\' + chr;
      } else if (exec(OTHER_PUNCTUATORS_AND_WHITESPACES, chr)) {
        result[i] = escapeChar(chr);
      } else {
        var charCode = charCodeAt(chr, 0);
        // single UTF-16 code unit
        if ((charCode & 0xF800) !== 0xD800) result[i] = chr;
        // unpaired surrogate
        else if (charCode >= 0xDC00 || i + 1 >= length || (charCodeAt(S, i + 1) & 0xFC00) !== 0xDC00) result[i] = escapeChar(chr);
        // surrogate pair
        else {
          result[i] = chr;
          result[++i] = charAt(S, i);
        }
      }
    }

    return join(result, '');
  }
});


/***/ },

/***/ 7642
(__unused_webpack_module, __unused_webpack_exports, __webpack_require__) {


var $ = __webpack_require__(6518);
var difference = __webpack_require__(3440);
var fails = __webpack_require__(9039);
var setMethodAcceptSetLike = __webpack_require__(4916);

var SET_LIKE_INCORRECT_BEHAVIOR = !setMethodAcceptSetLike('difference', function (result) {
  return result.size === 0;
});

var FORCED = SET_LIKE_INCORRECT_BEHAVIOR || fails(function () {
  // https://bugs.webkit.org/show_bug.cgi?id=288595
  var setLike = {
    size: 1,
    has: function () { return true; },
    keys: function () {
      var index = 0;
      return {
        next: function () {
          var done = index++ > 1;
          if (baseSet.has(1)) baseSet.clear();
          return { done: done, value: 2 };
        }
      };
    }
  };
  // eslint-disable-next-line es/no-set -- testing
  var baseSet = new Set([1, 2, 3, 4]);
  // eslint-disable-next-line es/no-set-prototype-difference -- testing
  return baseSet.difference(setLike).size !== 3;
});

// `Set.prototype.difference` method
// https://tc39.es/ecma262/#sec-set.prototype.difference
$({ target: 'Set', proto: true, real: true, forced: FORCED }, {
  difference: difference
});


/***/ },

/***/ 8004
(__unused_webpack_module, __unused_webpack_exports, __webpack_require__) {


var $ = __webpack_require__(6518);
var fails = __webpack_require__(9039);
var intersection = __webpack_require__(8750);
var setMethodAcceptSetLike = __webpack_require__(4916);

var INCORRECT = !setMethodAcceptSetLike('intersection', function (result) {
  return result.size === 2 && result.has(1) && result.has(2);
}) || fails(function () {
  // eslint-disable-next-line es/no-array-from, es/no-set, es/no-set-prototype-intersection -- testing
  return String(Array.from(new Set([1, 2, 3]).intersection(new Set([3, 2])))) !== '3,2';
});

// `Set.prototype.intersection` method
// https://tc39.es/ecma262/#sec-set.prototype.intersection
$({ target: 'Set', proto: true, real: true, forced: INCORRECT }, {
  intersection: intersection
});


/***/ },

/***/ 3853
(__unused_webpack_module, __unused_webpack_exports, __webpack_require__) {


var $ = __webpack_require__(6518);
var isDisjointFrom = __webpack_require__(4449);
var setMethodAcceptSetLike = __webpack_require__(4916);

var INCORRECT = !setMethodAcceptSetLike('isDisjointFrom', function (result) {
  return !result;
});

// `Set.prototype.isDisjointFrom` method
// https://tc39.es/ecma262/#sec-set.prototype.isdisjointfrom
$({ target: 'Set', proto: true, real: true, forced: INCORRECT }, {
  isDisjointFrom: isDisjointFrom
});


/***/ },

/***/ 5876
(__unused_webpack_module, __unused_webpack_exports, __webpack_require__) {


var $ = __webpack_require__(6518);
var isSubsetOf = __webpack_require__(3838);
var setMethodAcceptSetLike = __webpack_require__(4916);

var INCORRECT = !setMethodAcceptSetLike('isSubsetOf', function (result) {
  return result;
});

// `Set.prototype.isSubsetOf` method
// https://tc39.es/ecma262/#sec-set.prototype.issubsetof
$({ target: 'Set', proto: true, real: true, forced: INCORRECT }, {
  isSubsetOf: isSubsetOf
});


/***/ },

/***/ 2475
(__unused_webpack_module, __unused_webpack_exports, __webpack_require__) {


var $ = __webpack_require__(6518);
var isSupersetOf = __webpack_require__(8527);
var setMethodAcceptSetLike = __webpack_require__(4916);

var INCORRECT = !setMethodAcceptSetLike('isSupersetOf', function (result) {
  return !result;
});

// `Set.prototype.isSupersetOf` method
// https://tc39.es/ecma262/#sec-set.prototype.issupersetof
$({ target: 'Set', proto: true, real: true, forced: INCORRECT }, {
  isSupersetOf: isSupersetOf
});


/***/ },

/***/ 5024
(__unused_webpack_module, __unused_webpack_exports, __webpack_require__) {


var $ = __webpack_require__(6518);
var symmetricDifference = __webpack_require__(3650);
var setMethodGetKeysBeforeCloning = __webpack_require__(9835);
var setMethodAcceptSetLike = __webpack_require__(4916);

var FORCED = !setMethodAcceptSetLike('symmetricDifference') || !setMethodGetKeysBeforeCloning('symmetricDifference');

// `Set.prototype.symmetricDifference` method
// https://tc39.es/ecma262/#sec-set.prototype.symmetricdifference
$({ target: 'Set', proto: true, real: true, forced: FORCED }, {
  symmetricDifference: symmetricDifference
});


/***/ },

/***/ 1698
(__unused_webpack_module, __unused_webpack_exports, __webpack_require__) {


var $ = __webpack_require__(6518);
var union = __webpack_require__(4204);
var setMethodGetKeysBeforeCloning = __webpack_require__(9835);
var setMethodAcceptSetLike = __webpack_require__(4916);

var FORCED = !setMethodAcceptSetLike('union') || !setMethodGetKeysBeforeCloning('union');

// `Set.prototype.union` method
// https://tc39.es/ecma262/#sec-set.prototype.union
$({ target: 'Set', proto: true, real: true, forced: FORCED }, {
  union: union
});


/***/ },

/***/ 9577
(__unused_webpack_module, __unused_webpack_exports, __webpack_require__) {


var ArrayBufferViewCore = __webpack_require__(4644);
var isBigIntArray = __webpack_require__(1108);
var lengthOfArrayLike = __webpack_require__(6198);
var toIntegerOrInfinity = __webpack_require__(1291);
var toBigInt = __webpack_require__(5854);

var aTypedArray = ArrayBufferViewCore.aTypedArray;
var getTypedArrayConstructor = ArrayBufferViewCore.getTypedArrayConstructor;
var exportTypedArrayMethod = ArrayBufferViewCore.exportTypedArrayMethod;

var $RangeError = RangeError;

var PROPER_ORDER = function () {
  try {
    // eslint-disable-next-line no-throw-literal, es/no-typed-arrays, es/no-array-prototype-with -- required for testing
    new Int8Array(1)['with'](2, { valueOf: function () { throw 8; } });
  } catch (error) {
    // some early implementations, like WebKit, does not follow the final semantic
    // https://github.com/tc39/proposal-change-array-by-copy/pull/86
    return error === 8;
  }
}();

// Bug in WebKit. It should truncate a negative fractional index to zero, but instead throws an error
var THROW_ON_NEGATIVE_FRACTIONAL_INDEX = PROPER_ORDER && function () {
  try {
    // eslint-disable-next-line es/no-typed-arrays, es/no-array-prototype-with -- required for testing
    new Int8Array(1)['with'](-0.5, 1);
  } catch (error) {
    return true;
  }
}();

// `%TypedArray%.prototype.with` method
// https://tc39.es/ecma262/#sec-%typedarray%.prototype.with
exportTypedArrayMethod('with', { 'with': function (index, value) {
  var O = aTypedArray(this);
  var len = lengthOfArrayLike(O);
  var relativeIndex = toIntegerOrInfinity(index);
  var actualIndex = relativeIndex < 0 ? len + relativeIndex : relativeIndex;
  var numericValue = isBigIntArray(O) ? toBigInt(value) : +value;
  if (actualIndex >= len || actualIndex < 0) throw new $RangeError('Incorrect index');
  var A = new (getTypedArrayConstructor(O))(len);
  var k = 0;
  for (; k < len; k++) A[k] = k === actualIndex ? numericValue : O[k];
  return A;
} }['with'], !PROPER_ORDER || THROW_ON_NEGATIVE_FRACTIONAL_INDEX);


/***/ },

/***/ 9452
(__unused_webpack_module, __unused_webpack_exports, __webpack_require__) {


var $ = __webpack_require__(6518);
var aCallable = __webpack_require__(9306);
var aWeakMap = __webpack_require__(6557);
var aWeakKey = __webpack_require__(4328);
var WeakMapHelpers = __webpack_require__(4995);
var IS_PURE = __webpack_require__(6395);

var get = WeakMapHelpers.get;
var has = WeakMapHelpers.has;
var set = WeakMapHelpers.set;

var FORCED = IS_PURE || !function () {
  try {
    // eslint-disable-next-line es/no-weak-map, no-throw-literal -- testing
    if (WeakMap.prototype.getOrInsertComputed) new WeakMap().getOrInsertComputed(1, function () { throw 1; });
  } catch (error) {
    // FF144 Nightly - Beta 3 bug
    // https://bugzilla.mozilla.org/show_bug.cgi?id=1988369
    return error instanceof TypeError;
  }
}();

// `WeakMap.prototype.getOrInsertComputed` method
// https://tc39.es/ecma262/#sec-weakmap.prototype.getorinsertcomputed
$({ target: 'WeakMap', proto: true, real: true, forced: FORCED }, {
  getOrInsertComputed: function getOrInsertComputed(key, callbackfn) {
    if (!IS_PURE) aWeakMap(this);
    aWeakKey(key);
    aCallable(callbackfn);
    if (has(this, key)) return get(this, key);
    var value = callbackfn(key);
    set(this, key, value);
    return value;
  }
});


/***/ },

/***/ 8454
(__unused_webpack_module, __unused_webpack_exports, __webpack_require__) {


var $ = __webpack_require__(6518);
var WeakMapHelpers = __webpack_require__(4995);
var IS_PURE = __webpack_require__(6395);

var get = WeakMapHelpers.get;
var has = WeakMapHelpers.has;
var set = WeakMapHelpers.set;

// `WeakMap.prototype.getOrInsert` method
// https://tc39.es/ecma262/#sec-weakmap.prototype.getorinsert
$({ target: 'WeakMap', proto: true, real: true, forced: IS_PURE }, {
  getOrInsert: function getOrInsert(key, value) {
    if (has(this, key)) return get(this, key);
    set(this, key, value);
    return value;
  }
});


/***/ },

/***/ 5781
(__unused_webpack_module, __unused_webpack_exports, __webpack_require__) {


var $ = __webpack_require__(6518);
var getBuiltIn = __webpack_require__(7751);
var validateArgumentsLength = __webpack_require__(2812);
var toString = __webpack_require__(655);
var USE_NATIVE_URL = __webpack_require__(7416);

var URL = getBuiltIn('URL');

// `URL.parse` method
// https://url.spec.whatwg.org/#dom-url-parse
$({ target: 'URL', stat: true, forced: !USE_NATIVE_URL }, {
  parse: function parse(url) {
    var length = validateArgumentsLength(arguments.length, 1);
    var urlString = toString(url);
    var base = length < 2 || arguments[1] === undefined ? undefined : toString(arguments[1]);
    try {
      return new URL(urlString, base);
    } catch (error) {
      return null;
    }
  }
});


/***/ }

/******/ });
/************************************************************************/
/******/ // The module cache
/******/ const __webpack_module_cache__ = {};
/******/ 
/******/ // The require function
/******/ function __webpack_require__(moduleId) {
/******/ 	// Check if module is in cache
/******/ 	const cachedModule = __webpack_module_cache__[moduleId];
/******/ 	if (cachedModule !== undefined) {
/******/ 		return cachedModule.exports;
/******/ 	}
/******/ 	// Create a new module (and put it into the cache)
/******/ 	const module = __webpack_module_cache__[moduleId] = {
/******/ 		// no module.id needed
/******/ 		// no module.loaded needed
/******/ 		exports: {}
/******/ 	};
/******/ 
/******/ 	// Execute the module function
/******/ 	__webpack_modules__[moduleId].call(module.exports, module, module.exports, __webpack_require__);
/******/ 
/******/ 	// Return the exports of the module
/******/ 	return module.exports;
/******/ }
/******/ 
/************************************************************************/
/******/ /* webpack/runtime/define property getters */
/******/ (() => {
/******/ 	// define getter/value functions for harmony exports
/******/ 	__webpack_require__.d = (exports, definition) => {
/******/ 		if(Array.isArray(definition)) {
/******/ 			var i = 0;
/******/ 			while(i < definition.length) {
/******/ 				var key = definition[i++];
/******/ 				var binding = definition[i++];
/******/ 				if(!__webpack_require__.o(exports, key)) {
/******/ 					if(binding === 0) {
/******/ 						Object.defineProperty(exports, key, { enumerable: true, value: definition[i++] });
/******/ 					} else {
/******/ 						Object.defineProperty(exports, key, { enumerable: true, get: binding });
/******/ 					}
/******/ 				} else if(binding === 0) { i++; }
/******/ 			}
/******/ 		} else {
/******/ 			for(var key in definition) {
/******/ 				if(__webpack_require__.o(definition, key) && !__webpack_require__.o(exports, key)) {
/******/ 					Object.defineProperty(exports, key, { enumerable: true, get: definition[key] });
/******/ 				}
/******/ 			}
/******/ 		}
/******/ 	};
/******/ })();
/******/ 
/******/ /* webpack/runtime/hasOwnProperty shorthand */
/******/ (() => {
/******/ 	__webpack_require__.o = (obj, prop) => (Object.prototype.hasOwnProperty.call(obj, prop))
/******/ })();
/******/ 
/************************************************************************/

// EXTERNAL MODULE: ./node_modules/core-js/modules/es.array.push.js
var es_array_push = __webpack_require__(4114);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.iterator.constructor.js
var es_iterator_constructor = __webpack_require__(8111);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.iterator.filter.js
var es_iterator_filter = __webpack_require__(2489);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.iterator.for-each.js
var es_iterator_for_each = __webpack_require__(7588);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.iterator.map.js
var es_iterator_map = __webpack_require__(1701);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.json.stringify.js
var es_json_stringify = __webpack_require__(3110);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.map.get-or-insert.js
var es_map_get_or_insert = __webpack_require__(5367);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.map.get-or-insert-computed.js
var es_map_get_or_insert_computed = __webpack_require__(2731);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.regexp.escape.js
var es_regexp_escape = __webpack_require__(6069);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.set.difference.v2.js
var es_set_difference_v2 = __webpack_require__(7642);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.set.intersection.v2.js
var es_set_intersection_v2 = __webpack_require__(8004);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.set.is-disjoint-from.v2.js
var es_set_is_disjoint_from_v2 = __webpack_require__(3853);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.set.is-subset-of.v2.js
var es_set_is_subset_of_v2 = __webpack_require__(5876);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.set.is-superset-of.v2.js
var es_set_is_superset_of_v2 = __webpack_require__(2475);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.set.symmetric-difference.v2.js
var es_set_symmetric_difference_v2 = __webpack_require__(5024);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.set.union.v2.js
var es_set_union_v2 = __webpack_require__(1698);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.typed-array.with.js
var es_typed_array_with = __webpack_require__(9577);
;// ./web/pdfjs.js
const {
  AbortException,
  AnnotationEditorLayer,
  AnnotationEditorParamsType,
  AnnotationEditorType,
  AnnotationEditorUIManager,
  AnnotationLayer,
  AnnotationMode,
  AnnotationType,
  applyOpacity,
  build,
  ColorPicker,
  createValidAbsoluteUrl,
  CSSConstants,
  DOMSVGFactory,
  DrawLayer,
  FeatureTest,
  fetchData,
  findContrastColor,
  getDocument,
  getFilenameFromUrl,
  getPdfFilenameFromUrl,
  getRGB,
  getRGBA,
  getUuid,
  GlobalWorkerOptions,
  ImageKind,
  InvalidPDFException,
  isDataScheme,
  isPdfFile,
  isValidExplicitDest,
  makeArr,
  makeMap,
  makeObj,
  makeSet,
  MathClamp,
  noContextMenu,
  normalizeUnicode,
  OPS,
  OutputScale,
  PasswordException,
  PasswordResponses,
  PDFDataRangeTransport,
  PDFDateString,
  PDFWorker,
  PermissionFlag,
  PixelsPerInch,
  RenderingCancelledException,
  renderRichText,
  ResponseException,
  setLayerDimensions,
  shadow,
  SignatureExtractor,
  stopEvent,
  SupportedImageMimeTypes,
  TextLayer,
  TextLayerImages,
  TouchManager,
  updateUrlHash,
  Util,
  VerbosityLevel,
  version,
  XfaLayer
} = globalThis.pdfjsLib;

;// ./web/pdf_find_utils.js


let wordSegmenter = null;
let graphemeSegmenter = null;
function isWordBreakAt(content, pos) {
  graphemeSegmenter ||= new Intl.Segmenter(undefined, {
    granularity: "grapheme"
  });
  const graphemes = graphemeSegmenter.segment(content);
  const after = graphemes.containing(pos);
  if (after.index !== pos) {
    return false;
  }
  wordSegmenter ||= new Intl.Segmenter(undefined, {
    granularity: "word"
  });
  const before = graphemes.containing(pos - 1).segment;
  return wordSegmenter.segment(before + after.segment).containing(before.length).index === before.length;
}
function isEntireWord(content, startIdx, length) {
  const endIdx = startIdx + length;
  return (startIdx === 0 || isWordBreakAt(content, startIdx)) && (endIdx === content.length || isWordBreakAt(content, endIdx));
}
let NormalizeWithNFKC;
function getNormalizeWithNFKC() {
  if (!NormalizeWithNFKC) {
    const ranges = [];
    const range = [];
    const diacriticsRegex = /^\p{M}$/u;
    for (let i = 0; i < 65536; i++) {
      if (i >= 0xd800 && i <= 0xdfff) {
        continue;
      }
      const c = String.fromCharCode(i);
      if (c.normalize("NFKC") !== c && !diacriticsRegex.test(c)) {
        if (range.length !== 2) {
          range[0] = range[1] = i;
          continue;
        }
        if (range[1] + 1 !== i) {
          if (range[0] === range[1]) {
            ranges.push(String.fromCharCode(range[0]));
          } else {
            ranges.push(`${String.fromCharCode(range[0])}-${String.fromCharCode(range[1])}`);
          }
          range[0] = range[1] = i;
        } else {
          range[1] = i;
        }
      }
    }
    const rangesStr = ranges.join("");
    if (!NormalizeWithNFKC) {
      NormalizeWithNFKC = rangesStr;
    } else if (rangesStr !== NormalizeWithNFKC) {
      for (let i = 1; i < rangesStr.length; i++) {
        if (rangesStr[i] !== NormalizeWithNFKC[i]) {
          console.log(`Difference at index ${i}: ` + `U+${rangesStr.charCodeAt(i).toString(16).toUpperCase().padStart(4, "0")}` + `!== U+${NormalizeWithNFKC.charCodeAt(i).toString(16).toUpperCase().padStart(4, "0")}`);
          break;
        }
      }
      throw new Error("getNormalizeWithNFKC - update the `NormalizeWithNFKC` string.");
    }
  }
  return NormalizeWithNFKC;
}

// EXTERNAL MODULE: ./node_modules/core-js/modules/es.array.includes.js
var es_array_includes = __webpack_require__(4423);
;// ./web/ui_utils.js











const DEFAULT_SCALE_VALUE = "auto";
const DEFAULT_SCALE = 1.0;
const DEFAULT_SCALE_DELTA = 1.1;
const MIN_SCALE = 0.1;
const MAX_SCALE = 25.0;
const UNKNOWN_SCALE = 0;
const MAX_AUTO_SCALE = 1.25;
const SCROLLBAR_PADDING = 40;
const VERTICAL_PADDING = 5;
const PresentationModeState = {
  UNKNOWN: 0,
  NORMAL: 1,
  CHANGING: 2,
  FULLSCREEN: 3
};
const SidebarView = (/* unused pure expression or super */ null && ({
  UNKNOWN: -1,
  NONE: 0,
  THUMBS: 1,
  OUTLINE: 2,
  ATTACHMENTS: 3,
  LAYERS: 4
}));
const TextLayerMode = {
  DISABLE: 0,
  ENABLE: 1,
  ENABLE_PERMISSIONS: 2
};
const ScrollMode = {
  UNKNOWN: -1,
  VERTICAL: 0,
  HORIZONTAL: 1,
  WRAPPED: 2,
  PAGE: 3
};
const SpreadMode = {
  UNKNOWN: -1,
  NONE: 0,
  ODD: 1,
  EVEN: 2
};
const CursorTool = (/* unused pure expression or super */ null && ({
  SELECT: 0,
  HAND: 1,
  ZOOM: 2
}));
const AutoPrintRegExp = /\bprint\s*\(/;
function scrollIntoView(element, spot) {
  let parent = element.offsetParent;
  if (!parent) {
    console.error("offsetParent is not set -- cannot scroll");
    return;
  }
  let offsetY = element.offsetTop + element.clientTop;
  let offsetX = element.offsetLeft + element.clientLeft;
  while (parent.clientHeight === parent.scrollHeight && parent.clientWidth === parent.scrollWidth) {
    offsetY += parent.offsetTop;
    offsetX += parent.offsetLeft;
    parent = parent.offsetParent;
    if (!parent) {
      return;
    }
  }
  if (spot) {
    if (spot.top !== undefined) {
      offsetY += spot.top;
    }
    if (spot.left !== undefined) {
      offsetX += spot.left;
      parent.scrollLeft = offsetX;
    }
  }
  parent.scrollTop = offsetY;
}
function watchScroll(viewAreaElement, callback, abortSignal = undefined) {
  function onRAF() {
    rAF = null;
    const currentX = viewAreaElement.scrollLeft;
    const lastX = state.lastX;
    if (currentX !== lastX) {
      state.right = currentX > lastX;
    }
    state.lastX = currentX;
    const currentY = viewAreaElement.scrollTop;
    const lastY = state.lastY;
    if (currentY !== lastY) {
      state.down = currentY > lastY;
    }
    state.lastY = currentY;
    callback(state);
  }
  const state = {
    right: true,
    down: true,
    lastX: viewAreaElement.scrollLeft,
    lastY: viewAreaElement.scrollTop
  };
  let rAF = null;
  viewAreaElement.addEventListener("scroll", () => {
    rAF ??= window.requestAnimationFrame(onRAF);
  }, {
    useCapture: true,
    signal: abortSignal
  });
  abortSignal?.addEventListener("abort", () => window.cancelAnimationFrame(rAF), {
    once: true
  });
  return state;
}
function parseQueryString(query) {
  const params = new Map();
  for (const [key, value] of new URLSearchParams(query)) {
    params.set(key.toLowerCase(), value);
  }
  return params;
}
const InvisibleCharsRegExp = /[\x00-\x1F]/g;
function removeNullCharacters(str, replaceInvisible = false) {
  if (!InvisibleCharsRegExp.test(str)) {
    return str;
  }
  if (replaceInvisible) {
    return str.replaceAll(InvisibleCharsRegExp, m => m === "\x00" ? "" : " ");
  }
  return str.replaceAll("\x00", "");
}
function binarySearchFirstItem(items, condition, start = 0) {
  let minIndex = start;
  let maxIndex = items.length - 1;
  if (maxIndex < 0 || !condition(items[maxIndex])) {
    return items.length;
  }
  if (condition(items[minIndex])) {
    return minIndex;
  }
  while (minIndex < maxIndex) {
    const currentIndex = minIndex + maxIndex >> 1;
    const currentItem = items[currentIndex];
    if (condition(currentItem)) {
      maxIndex = currentIndex;
    } else {
      minIndex = currentIndex + 1;
    }
  }
  return minIndex;
}
function approximateFraction(x) {
  if (Math.floor(x) === x) {
    return [x, 1];
  }
  const xinv = 1 / x;
  const limit = 8;
  if (xinv > limit) {
    return [1, limit];
  } else if (Math.floor(xinv) === xinv) {
    return [1, xinv];
  }
  const x_ = x > 1 ? xinv : x;
  let a = 0,
    b = 1,
    c = 1,
    d = 1;
  while (true) {
    const p = a + c,
      q = b + d;
    if (q > limit) {
      break;
    }
    if (x_ <= p / q) {
      c = p;
      d = q;
    } else {
      a = p;
      b = q;
    }
  }
  if (x_ - a / b < c / d - x_) {
    return x_ === x ? [a, b] : [b, a];
  }
  return x_ === x ? [c, d] : [d, c];
}
function floorToDivide(x, div) {
  return x - x % div;
}
function getPageSizeInches({
  view,
  userUnit,
  rotate
}) {
  const [x1, y1, x2, y2] = view;
  const changeOrientation = rotate % 180 !== 0;
  const width = (x2 - x1) / 72 * userUnit;
  const height = (y2 - y1) / 72 * userUnit;
  return {
    width: changeOrientation ? height : width,
    height: changeOrientation ? width : height
  };
}
function backtrackBeforeAllVisibleElements(index, views, top) {
  if (index < 2) {
    return index;
  }
  let elt = views[index].div;
  let pageTop = elt.offsetTop + elt.clientTop;
  if (pageTop >= top) {
    elt = views[index - 1].div;
    pageTop = elt.offsetTop + elt.clientTop;
  }
  for (let i = index - 2; i >= 0; --i) {
    elt = views[i].div;
    if (elt.offsetTop + elt.clientTop + elt.clientHeight <= pageTop) {
      break;
    }
    index = i;
  }
  return index;
}
function visibleSort(a, b) {
  const pc = a.percent - b.percent;
  return Math.abs(pc) > 0.001 ? -pc : a.id - b.id;
}
function getVisibleElements({
  scrollEl,
  views,
  sortByVisibility = false,
  horizontal = false,
  rtl = false
}) {
  const top = scrollEl.scrollTop,
    bottom = top + scrollEl.clientHeight;
  const left = scrollEl.scrollLeft,
    right = left + scrollEl.clientWidth;
  function isElementBottomAfterViewTop(view) {
    const element = view.div;
    const elementBottom = element.offsetTop + element.clientTop + element.clientHeight;
    return elementBottom > top;
  }
  function isElementNextAfterViewHorizontally(view) {
    const element = view.div;
    const elementLeft = element.offsetLeft + element.clientLeft;
    const elementRight = elementLeft + element.clientWidth;
    return rtl ? elementLeft < right : elementRight > left;
  }
  const visible = [],
    ids = new Set(),
    numViews = views.length;
  let firstVisibleElementInd = binarySearchFirstItem(views, horizontal ? isElementNextAfterViewHorizontally : isElementBottomAfterViewTop);
  if (firstVisibleElementInd > 0 && firstVisibleElementInd < numViews && !horizontal) {
    firstVisibleElementInd = backtrackBeforeAllVisibleElements(firstVisibleElementInd, views, top);
  }
  let lastEdge = horizontal ? right : -1;
  for (let i = firstVisibleElementInd; i < numViews; i++) {
    const view = views[i],
      element = view.div;
    const currentWidth = element.offsetLeft + element.clientLeft;
    const currentHeight = element.offsetTop + element.clientTop;
    const viewWidth = element.clientWidth,
      viewHeight = element.clientHeight;
    const viewRight = currentWidth + viewWidth;
    const viewBottom = currentHeight + viewHeight;
    if (lastEdge === -1) {
      if (viewBottom >= bottom) {
        lastEdge = viewBottom;
      }
    } else if ((horizontal ? currentWidth : currentHeight) > lastEdge) {
      break;
    }
    if (viewBottom <= top || currentHeight >= bottom || viewRight <= left || currentWidth >= right) {
      continue;
    }
    const minY = Math.max(0, top - currentHeight);
    const minX = Math.max(0, left - currentWidth);
    const hiddenHeight = minY + Math.max(0, viewBottom - bottom);
    const hiddenWidth = minX + Math.max(0, viewRight - right);
    const fractionHeight = (viewHeight - hiddenHeight) / viewHeight,
      fractionWidth = (viewWidth - hiddenWidth) / viewWidth;
    const percent = fractionHeight * fractionWidth * 100 | 0;
    visible.push({
      id: view.id,
      x: currentWidth,
      y: currentHeight,
      visibleArea: percent === 100 ? null : {
        minX,
        minY,
        maxX: Math.min(viewRight, right) - currentWidth,
        maxY: Math.min(viewBottom, bottom) - currentHeight
      },
      view,
      percent,
      widthPercent: fractionWidth * 100 | 0
    });
    ids.add(view.id);
  }
  const first = visible[0],
    last = visible.at(-1);
  if (sortByVisibility) {
    visible.sort(visibleSort);
  }
  return {
    first,
    last,
    views: visible,
    ids
  };
}
function normalizeWheelEventDirection(evt) {
  let delta = Math.hypot(evt.deltaX, evt.deltaY);
  const angle = Math.atan2(evt.deltaY, evt.deltaX);
  if (-0.25 * Math.PI < angle && angle < 0.75 * Math.PI) {
    delta = -delta;
  }
  return delta;
}
function normalizeWheelEventDelta(evt) {
  const deltaMode = evt.deltaMode;
  let delta = normalizeWheelEventDirection(evt);
  const MOUSE_PIXELS_PER_LINE = 30;
  const MOUSE_LINES_PER_PAGE = 30;
  if (deltaMode === WheelEvent.DOM_DELTA_PIXEL) {
    delta /= MOUSE_PIXELS_PER_LINE * MOUSE_LINES_PER_PAGE;
  } else if (deltaMode === WheelEvent.DOM_DELTA_LINE) {
    delta /= MOUSE_LINES_PER_PAGE;
  }
  return delta;
}
function isValidRotation(angle) {
  return Number.isInteger(angle) && angle % 90 === 0;
}
function isValidScrollMode(mode) {
  return Number.isInteger(mode) && Object.values(ScrollMode).includes(mode) && mode !== ScrollMode.UNKNOWN;
}
function isValidSpreadMode(mode) {
  return Number.isInteger(mode) && Object.values(SpreadMode).includes(mode) && mode !== SpreadMode.UNKNOWN;
}
function isPortraitOrientation(size) {
  return size.width <= size.height;
}
const animationStarted = new Promise(function (resolve) {
  window.requestAnimationFrame(resolve);
});
const docStyle = document.documentElement.style;
class ProgressBar {
  #classList = null;
  #disableAutoFetchTimeout = null;
  #percent = 0;
  #style = null;
  #visible = true;
  constructor(bar) {
    this.#classList = bar.classList;
    this.#style = bar.style;
  }
  get percent() {
    return this.#percent;
  }
  set percent(val) {
    this.#percent = val;
    if (isNaN(val)) {
      this.#classList.add("indeterminate");
      return;
    }
    this.#classList.remove("indeterminate");
    this.#style.setProperty("--progressBar-percent", `${this.#percent}%`);
  }
  setWidth(viewer) {
    if (!viewer) {
      return;
    }
    const container = viewer.parentNode;
    const scrollbarWidth = container.offsetWidth - viewer.offsetWidth;
    if (scrollbarWidth > 0) {
      this.#style.setProperty("--progressBar-end-offset", `${scrollbarWidth}px`);
    }
  }
  setDisableAutoFetch(delay = 5000) {
    if (this.#percent === 100 || isNaN(this.#percent)) {
      return;
    }
    if (this.#disableAutoFetchTimeout) {
      clearTimeout(this.#disableAutoFetchTimeout);
    }
    this.show();
    this.#disableAutoFetchTimeout = setTimeout(() => {
      this.#disableAutoFetchTimeout = null;
      this.hide();
    }, delay);
  }
  hide() {
    if (!this.#visible) {
      return;
    }
    this.#visible = false;
    this.#classList.add("hidden");
  }
  show() {
    if (this.#visible) {
      return;
    }
    this.#visible = true;
    this.#classList.remove("hidden");
  }
}
function getActiveOrFocusedElement() {
  let curRoot = document;
  let curActiveOrFocused = curRoot.activeElement || curRoot.querySelector(":focus");
  while (curActiveOrFocused?.shadowRoot) {
    curRoot = curActiveOrFocused.shadowRoot;
    curActiveOrFocused = curRoot.activeElement || curRoot.querySelector(":focus");
  }
  return curActiveOrFocused;
}
function apiPageLayoutToViewerModes(layout) {
  let scrollMode = ScrollMode.VERTICAL,
    spreadMode = SpreadMode.NONE;
  switch (layout) {
    case "SinglePage":
      scrollMode = ScrollMode.PAGE;
      break;
    case "OneColumn":
      break;
    case "TwoPageLeft":
      scrollMode = ScrollMode.PAGE;
    case "TwoColumnLeft":
      spreadMode = SpreadMode.ODD;
      break;
    case "TwoPageRight":
      scrollMode = ScrollMode.PAGE;
    case "TwoColumnRight":
      spreadMode = SpreadMode.EVEN;
      break;
  }
  return {
    scrollMode,
    spreadMode
  };
}
function apiPageModeToSidebarView(mode) {
  switch (mode) {
    case "UseNone":
      return SidebarView.NONE;
    case "UseThumbs":
      return SidebarView.THUMBS;
    case "UseOutlines":
      return SidebarView.OUTLINE;
    case "UseAttachments":
      return SidebarView.ATTACHMENTS;
    case "UseOC":
      return SidebarView.LAYERS;
  }
  return SidebarView.NONE;
}
function toggleCheckedBtn(button, toggle, view = null) {
  button.classList.toggle("toggled", toggle);
  button.setAttribute("aria-checked", toggle);
  view?.classList.toggle("hidden", !toggle);
}
function toggleSelectedBtn(button, toggle, view = null) {
  button.classList.toggle("selected", toggle);
  button.setAttribute("aria-selected", toggle);
  view?.classList.toggle("hidden", !toggle);
}
function toggleExpandedBtn(button, toggle, view = null) {
  button.classList.toggle("toggled", toggle);
  button.setAttribute("aria-expanded", toggle);
  view?.classList.toggle("hidden", !toggle);
}
const calcRound = function () {
  const e = document.createElement("div");
  e.style.width = "round(down, calc(1.6666666666666665 * 792px), 1px)";
  return e.style.width === "calc(1320px)" ? Math.fround : x => x;
}();

;// ./web/internal_evt.js
const INTERNAL_EVT = "59968104-cc61-4cf9-b570-014b35b3709c";
const internalOpt = Object.freeze({
  internal: INTERNAL_EVT
});

;// ./web/pdf_find_controller.js




















const FindState = {
  FOUND: 0,
  NOT_FOUND: 1,
  WRAPPED: 2,
  PENDING: 3
};
const CHARACTERS_TO_NORMALIZE = new Map([["\u2010", "-"], ["\u2018", "'"], ["\u2019", "'"], ["\u201A", "'"], ["\u201B", "'"], ["\u201C", '"'], ["\u201D", '"'], ["\u201E", '"'], ["\u201F", '"'], ["\u00BC", "1/4"], ["\u00BD", "1/2"], ["\u00BE", "3/4"]]);
const DIACRITICS_EXCEPTION = new Set([0x3099, 0x309a, 0x094d, 0x09cd, 0x0a4d, 0x0acd, 0x0b4d, 0x0bcd, 0x0c4d, 0x0ccd, 0x0d3b, 0x0d3c, 0x0d4d, 0x0dca, 0x0e3a, 0x0eba, 0x0f84, 0x1039, 0x103a, 0x1714, 0x1734, 0x17d2, 0x1a60, 0x1b44, 0x1baa, 0x1bab, 0x1bf2, 0x1bf3, 0x2d7f, 0xa806, 0xa82c, 0xa8c4, 0xa953, 0xa9c0, 0xaaf6, 0xabed, 0x0c56, 0x0f71, 0x0f72, 0x0f7a, 0x0f7b, 0x0f7c, 0x0f7d, 0x0f80, 0x0f74]);
let DIACRITICS_EXCEPTION_STR;
const DIACRITICS_REG_EXP = /\p{M}+/gu;
const SPECIAL_CHARS_REG_EXP = /([+^$|])|(\p{P}+)|(\s+)|(\p{M})|(\p{L})/gu;
const SYLLABLES_REG_EXP = /[\uAC00-\uD7AF\uFA6C\uFACF-\uFAD1\uFAD5-\uFAD7]+/g;
const SYLLABLES_LENGTHS = new Map();
const FIRST_CHAR_SYLLABLES_REG_EXP = "[\\u1100-\\u1112\\ud7a4-\\ud7af\\ud84a\\ud84c\\ud850\\ud854\\ud857\\ud85f]";
const NFKC_CHARS_TO_NORMALIZE = new Map();
let noSyllablesRegExp = null;
let withSyllablesRegExp = null;
function normalize(text, options = {}) {
  const syllablePositions = [];
  let m;
  while ((m = SYLLABLES_REG_EXP.exec(text)) !== null) {
    let {
      index
    } = m;
    for (const char of m[0]) {
      let len = SYLLABLES_LENGTHS.get(char);
      if (!len) {
        len = char.normalize("NFD").length;
        SYLLABLES_LENGTHS.set(char, len);
      }
      syllablePositions.push([len, index++]);
    }
  }
  const hasSyllables = syllablePositions.length > 0;
  const ignoreDashEOL = options.ignoreDashEOL ?? false;
  let normalizationRegex;
  if (!hasSyllables && noSyllablesRegExp) {
    normalizationRegex = noSyllablesRegExp;
  } else if (hasSyllables && withSyllablesRegExp) {
    normalizationRegex = withSyllablesRegExp;
  } else {
    const replace = CHARACTERS_TO_NORMALIZE.keys().join("");
    const toNormalizeWithNFKC = getNormalizeWithNFKC();
    const CJK = "(?:\\p{Ideographic}|[\u3040-\u30FF])";
    const HKDiacritics = "(?:\u3099|\u309A)";
    const BrokenWord = `\\p{Ll}-\\n(?=\\p{Ll})|\\p{Lu}-\\n(?=\\p{L})`;
    const regexps = [`[${replace}]`, `[${toNormalizeWithNFKC}]`, `${HKDiacritics}\\n`, "\\p{M}+(?:-\\n)?", `${BrokenWord}`, "\\S-\\n", `${CJK}\\n`, "\\n", hasSyllables ? FIRST_CHAR_SYLLABLES_REG_EXP : "\\u0000"];
    normalizationRegex = new RegExp(regexps.map(r => `(${r})`).join("|"), "gmu");
    if (hasSyllables) {
      withSyllablesRegExp = normalizationRegex;
    } else {
      noSyllablesRegExp = normalizationRegex;
    }
  }
  const rawDiacriticsPositions = [];
  while ((m = DIACRITICS_REG_EXP.exec(text)) !== null) {
    rawDiacriticsPositions.push([m[0].length, m.index]);
  }
  let normalized = text.normalize("NFD");
  const positions = [0, 0];
  let rawDiacriticsIndex = 0;
  let syllableIndex = 0;
  let shift = 0;
  let shiftOrigin = 0;
  let eol = 0;
  let hasDiacritics = false;
  normalized = normalized.replace(normalizationRegex, (match, p1, p2, p3, p4, p5, p6, p7, p8, p9, i) => {
    i -= shiftOrigin;
    if (p1) {
      const replacement = CHARACTERS_TO_NORMALIZE.get(p1);
      const jj = replacement.length;
      for (let j = 1; j < jj; j++) {
        positions.push(i - shift + j, shift - j);
      }
      shift -= jj - 1;
      return replacement;
    }
    if (p2) {
      const replacement = NFKC_CHARS_TO_NORMALIZE.getOrInsertComputed(p2, () => p2.normalize("NFKC"));
      const jj = replacement.length;
      for (let j = 1; j < jj; j++) {
        positions.push(i - shift + j, shift - j);
      }
      shift -= jj - 1;
      return replacement;
    }
    if (p3) {
      hasDiacritics = true;
      if (i + eol === rawDiacriticsPositions[rawDiacriticsIndex]?.[1]) {
        ++rawDiacriticsIndex;
      } else {
        positions.push(i - 1 - shift + 1, shift - 1);
        shift -= 1;
        shiftOrigin += 1;
      }
      positions.push(i - shift + 1, shift);
      shiftOrigin += 1;
      eol += 1;
      return p3.charAt(0);
    }
    if (p4) {
      const hasTrailingDashEOL = p4.endsWith("\n");
      const len = hasTrailingDashEOL ? p4.length - 2 : p4.length;
      hasDiacritics = true;
      let jj = len;
      if (i + eol === rawDiacriticsPositions[rawDiacriticsIndex]?.[1]) {
        jj -= rawDiacriticsPositions[rawDiacriticsIndex][0];
        ++rawDiacriticsIndex;
      }
      for (let j = 1; j <= jj; j++) {
        positions.push(i - 1 - shift + j, shift - j);
      }
      shift -= jj;
      shiftOrigin += jj;
      if (hasTrailingDashEOL) {
        i += len - 1;
        positions.push(i - shift + 1, 1 + shift);
        shift += 1;
        shiftOrigin += 1;
        eol += 1;
        return p4.slice(0, len);
      }
      return p4;
    }
    if (p5) {
      if (ignoreDashEOL) {
        shiftOrigin += 1;
        eol += 1;
        return p5.slice(0, -1);
      }
      const len = p5.length - 2;
      positions.push(i - shift + len, 1 + shift);
      shift += 1;
      shiftOrigin += 1;
      eol += 1;
      return p5.slice(0, -2);
    }
    if (p6) {
      shiftOrigin += 1;
      eol += 1;
      return p6.slice(0, -1);
    }
    if (p7) {
      const len = p7.length - 1;
      positions.push(i - shift + len, shift);
      shiftOrigin += 1;
      eol += 1;
      return p7.slice(0, -1);
    }
    if (p8) {
      positions.push(i - shift + 1, shift - 1);
      shift -= 1;
      shiftOrigin += 1;
      eol += 1;
      return " ";
    }
    if (i + eol === syllablePositions[syllableIndex]?.[1]) {
      const newCharLen = syllablePositions[syllableIndex][0] - 1;
      ++syllableIndex;
      for (let j = 1; j <= newCharLen; j++) {
        positions.push(i - (shift - j), shift - j);
      }
      shift -= newCharLen;
      shiftOrigin += newCharLen;
    }
    return p9;
  });
  positions.push(normalized.length, shift);
  const starts = new Uint32Array(positions.length >> 1);
  const shifts = new Int32Array(positions.length >> 1);
  for (let i = 0, ii = positions.length; i < ii; i += 2) {
    starts[i >> 1] = positions[i];
    shifts[i >> 1] = positions[i + 1];
  }
  return [normalized, [starts, shifts], hasDiacritics];
}
function getOriginalIndex(diffs, pos, len) {
  if (!diffs) {
    return [pos, len];
  }
  const [starts, shifts] = diffs;
  const start = pos;
  const end = pos + len - 1;
  let i = binarySearchFirstItem(starts, x => x >= start);
  if (starts[i] > start) {
    --i;
  }
  let j = binarySearchFirstItem(starts, x => x >= end, i);
  if (starts[j] > end) {
    --j;
  }
  const oldStart = start + shifts[i];
  const oldEnd = end + shifts[j];
  const oldLen = oldEnd + 1 - oldStart;
  return [oldStart, oldLen];
}
class PDFFindController {
  #state = null;
  #updateMatchesCountOnProgress = true;
  #delay = 0;
  #visitedPagesCount = 0;
  #copiedPageData = null;
  #savedPageData = null;
  constructor({
    linkService,
    eventBus,
    delay = 250,
    updateMatchesCountOnProgress = true
  }) {
    this._linkService = linkService;
    this._eventBus = eventBus;
    this.#updateMatchesCountOnProgress = updateMatchesCountOnProgress;
    this.#delay = delay;
    this.onIsPageVisible = null;
    this.#reset();
    eventBus.on("find", this.#onFind.bind(this), internalOpt);
    eventBus.on("findbarclose", this.#onFindBarClose.bind(this), internalOpt);
    eventBus.on("pagesedited", this.#onPagesEdited.bind(this), internalOpt);
  }
  get highlightMatches() {
    return this._highlightMatches;
  }
  get pageMatches() {
    return this._pageMatches;
  }
  get pageMatchesLength() {
    return this._pageMatchesLength;
  }
  get selected() {
    return this._selected;
  }
  get state() {
    return this.#state;
  }
  setDocument(pdfDocument) {
    if (this._pdfDocument) {
      this.#reset();
    }
    if (!pdfDocument) {
      return;
    }
    this._pdfDocument = pdfDocument;
    this._firstPageCapability.resolve();
  }
  #onFind(state) {
    if (!state) {
      return;
    }
    const pdfDocument = this._pdfDocument;
    const {
      type
    } = state;
    if (this.#state === null || this.#shouldDirtyMatch(state)) {
      this._dirtyMatch = true;
    }
    this.#state = state;
    if (type !== "highlightallchange") {
      this.#updateUIState(FindState.PENDING);
    }
    this._firstPageCapability.promise.then(() => {
      if (!this._pdfDocument || pdfDocument && this._pdfDocument !== pdfDocument) {
        return;
      }
      this.#extractText();
      const findbarClosed = !this._highlightMatches;
      const pendingTimeout = !!this._findTimeout;
      if (this._findTimeout) {
        clearTimeout(this._findTimeout);
        this._findTimeout = null;
      }
      if (!type) {
        this._findTimeout = setTimeout(() => {
          this.#nextMatch();
          this._findTimeout = null;
        }, this.#delay);
      } else if (this._dirtyMatch) {
        this.#nextMatch();
      } else if (type === "again") {
        this.#nextMatch();
        if (findbarClosed && this.#state.highlightAll) {
          this.#updateAllPages();
        }
      } else if (type === "highlightallchange") {
        if (pendingTimeout) {
          this.#nextMatch();
        } else {
          this._highlightMatches = true;
        }
        this.#updateAllPages();
      } else {
        this.#nextMatch();
      }
    });
  }
  scrollMatchIntoView({
    element = null,
    pageIndex = -1,
    matchIndex = -1
  }) {
    if (!this._scrollMatches || !element) {
      return;
    } else if (matchIndex === -1 || matchIndex !== this._selected.matchIdx) {
      return;
    } else if (pageIndex === -1 || pageIndex !== this._selected.pageIdx) {
      return;
    }
    this._scrollMatches = false;
    element.scrollIntoView({
      block: "start",
      inline: "center"
    });
  }
  #reset() {
    this._highlightMatches = false;
    this._scrollMatches = false;
    this._pdfDocument = null;
    this._pageMatches = [];
    this._pageMatchesLength = [];
    this.#visitedPagesCount = 0;
    this.#state = null;
    this._selected = {
      pageIdx: -1,
      matchIdx: -1
    };
    this._offset = {
      pageIdx: null,
      matchIdx: null,
      wrapped: false
    };
    this._extractTextPromises = [];
    this._pageContents = [];
    this._pageDiffs = [];
    this._hasDiacritics = [];
    this._matchesCountTotal = 0;
    this._pagesToSearch = null;
    this._pendingFindMatches = new Set();
    this._resumePageIdx = null;
    this._dirtyMatch = false;
    clearTimeout(this._findTimeout);
    this._findTimeout = null;
    this.#copiedPageData = null;
    this._firstPageCapability = Promise.withResolvers();
  }
  get #query() {
    const {
      query
    } = this.#state;
    if (typeof query === "string") {
      if (query !== this._rawQuery) {
        this._rawQuery = query;
        [this._normalizedQuery] = normalize(query);
      }
      return this._normalizedQuery;
    }
    return (query || []).filter(Boolean).map(q => normalize(q)[0]);
  }
  #shouldDirtyMatch(state) {
    const newQuery = state.query,
      prevQuery = this.#state.query;
    const newType = typeof newQuery,
      prevType = typeof prevQuery;
    if (newType !== prevType) {
      return true;
    }
    if (newType === "string") {
      if (newQuery !== prevQuery) {
        return true;
      }
    } else if (JSON.stringify(newQuery) !== JSON.stringify(prevQuery)) {
      return true;
    }
    switch (state.type) {
      case "again":
        const pageNumber = this._selected.pageIdx + 1;
        const linkService = this._linkService;
        return pageNumber >= 1 && pageNumber <= linkService.pagesCount && pageNumber !== linkService.page && !(this.onIsPageVisible?.(pageNumber) ?? true);
      case "highlightallchange":
        return false;
    }
    return true;
  }
  #convertToRegExpString(query, hasDiacritics) {
    const {
      matchDiacritics
    } = this.#state;
    let isUnicode = false;
    const addExtraWhitespaces = (original, fixed) => {
      if (original === query) {
        return fixed;
      }
      if (query.startsWith(original)) {
        return `${fixed}[ ]*`;
      }
      if (query.endsWith(original)) {
        return `[ ]*${fixed}`;
      }
      return `[ ]*${fixed}[ ]*`;
    };
    query = query.replaceAll(SPECIAL_CHARS_REG_EXP, (match, p1, p2, p3, p4, p5) => {
      if (p1) {
        return addExtraWhitespaces(p1, RegExp.escape(p1));
      }
      if (p2) {
        return addExtraWhitespaces(p2, RegExp.escape(p2));
      }
      if (p3) {
        return "[ ]+";
      }
      if (matchDiacritics) {
        return p4 || p5;
      }
      if (p4) {
        return DIACRITICS_EXCEPTION.has(p4.charCodeAt(0)) ? p4 : "";
      }
      if (hasDiacritics) {
        isUnicode = true;
        return `${p5}\\p{M}*`;
      }
      return p5;
    });
    const trailingSpaces = "[ ]*";
    if (query.endsWith(trailingSpaces)) {
      query = query.slice(0, query.length - trailingSpaces.length);
    }
    if (matchDiacritics) {
      if (hasDiacritics) {
        DIACRITICS_EXCEPTION_STR ||= String.fromCharCode(...DIACRITICS_EXCEPTION);
        isUnicode = true;
        query = `${query}(?=[${DIACRITICS_EXCEPTION_STR}]|[^\\p{M}]|$)`;
      }
    }
    return [isUnicode, query];
  }
  #calculateMatch(pageIndex) {
    if (!this.#state) {
      return;
    }
    const query = this.#query;
    if (query.length === 0) {
      return;
    }
    const pageContent = this._pageContents[pageIndex];
    const matcherResult = this.match(query, pageContent, pageIndex);
    const matches = this._pageMatches[pageIndex] = [];
    const matchesLength = this._pageMatchesLength[pageIndex] = [];
    const diffs = this._pageDiffs[pageIndex];
    matcherResult?.forEach(({
      index,
      length
    }) => {
      const [matchPos, matchLen] = getOriginalIndex(diffs, index, length);
      if (matchLen) {
        matches.push(matchPos);
        matchesLength.push(matchLen);
      }
    });
    if (this.#state.highlightAll) {
      this.#updatePage(pageIndex);
    }
    if (this._resumePageIdx === pageIndex) {
      this._resumePageIdx = null;
      this.#nextPageMatch();
    }
    const pageMatchesCount = matches.length;
    this._matchesCountTotal += pageMatchesCount;
    if (this.#updateMatchesCountOnProgress) {
      if (pageMatchesCount > 0) {
        this.#updateUIResultsCount();
      }
    } else if (++this.#visitedPagesCount === this._linkService.pagesCount) {
      this.#updateUIResultsCount();
    }
  }
  match(query, pageContent, pageIndex) {
    const hasDiacritics = this._hasDiacritics[pageIndex];
    let isUnicode = false;
    if (typeof query === "string") {
      [isUnicode, query] = this.#convertToRegExpString(query, hasDiacritics);
    } else {
      query = query.sort().reverse().map(q => {
        const [isUnicodePart, queryPart] = this.#convertToRegExpString(q, hasDiacritics);
        isUnicode ||= isUnicodePart;
        return `(${queryPart})`;
      }).join("|");
    }
    if (!query) {
      return undefined;
    }
    const {
      caseSensitive,
      entireWord
    } = this.#state;
    const flags = `g${isUnicode ? "u" : ""}${caseSensitive ? "" : "i"}`;
    query = new RegExp(query, flags);
    const matches = [];
    let match;
    while ((match = query.exec(pageContent)) !== null) {
      if (entireWord && !isEntireWord(pageContent, match.index, match[0].length)) {
        continue;
      }
      matches.push({
        index: match.index,
        length: match[0].length
      });
    }
    return matches;
  }
  #extractText() {
    if (this._extractTextPromises.length > 0) {
      return;
    }
    let deferred = Promise.resolve();
    const textOptions = {
      disableNormalization: true
    };
    const pdfDoc = this._pdfDocument;
    for (let i = 0, ii = this._linkService.pagesCount; i < ii; i++) {
      const {
        promise,
        resolve
      } = Promise.withResolvers();
      this._extractTextPromises[i] = promise;
      deferred = deferred.then(async () => {
        if (pdfDoc !== this._pdfDocument) {
          resolve();
          return;
        }
        try {
          const pdfPage = await pdfDoc.getPage(i + 1);
          const textContent = await pdfPage.getTextContent(textOptions);
          if (pdfDoc !== this._pdfDocument) {
            resolve();
            return;
          }
          const strBuf = [];
          for (const textItem of textContent.items) {
            strBuf.push(textItem.str);
            if (textItem.hasEOL) {
              strBuf.push("\n");
            }
          }
          [this._pageContents[i], this._pageDiffs[i], this._hasDiacritics[i]] = normalize(strBuf.join(""));
        } catch (ex) {
          if (pdfDoc !== this._pdfDocument) {
            resolve();
            return;
          }
          console.error(`Unable to get text content for page ${i + 1}`, ex);
          [this._pageContents[i], this._pageDiffs[i], this._hasDiacritics[i]] = ["", null, false];
        }
        resolve();
      });
    }
  }
  #updatePage(index) {
    if (this._scrollMatches && this._selected.pageIdx === index) {
      this._linkService.page = index + 1;
    }
    this._eventBus.dispatch("updatetextlayermatches", {
      source: this,
      pageIndex: index
    });
  }
  #updateAllPages() {
    this._eventBus.dispatch("updatetextlayermatches", {
      source: this,
      pageIndex: -1
    });
  }
  #nextMatch() {
    const previous = this.#state.findPrevious;
    const currentPageIndex = this._linkService.page - 1;
    const numPages = this._linkService.pagesCount;
    this._highlightMatches = true;
    if (this._dirtyMatch) {
      this._dirtyMatch = false;
      this._selected.pageIdx = this._selected.matchIdx = -1;
      this._offset.pageIdx = currentPageIndex;
      this._offset.matchIdx = null;
      this._offset.wrapped = false;
      this._resumePageIdx = null;
      this._pageMatches.length = 0;
      this._pageMatchesLength.length = 0;
      this.#visitedPagesCount = 0;
      this._matchesCountTotal = 0;
      this.#updateAllPages();
      for (let i = 0; i < numPages; i++) {
        if (this._pendingFindMatches.has(i)) {
          continue;
        }
        this._pendingFindMatches.add(i);
        this._extractTextPromises[i].then(() => {
          this._pendingFindMatches.delete(i);
          this.#calculateMatch(i);
        });
      }
    }
    const query = this.#query;
    if (query.length === 0) {
      this.#updateUIState(FindState.FOUND);
      return;
    }
    if (this._resumePageIdx) {
      return;
    }
    const offset = this._offset;
    this._pagesToSearch = numPages;
    if (offset.matchIdx !== null) {
      const numPageMatches = this._pageMatches[offset.pageIdx].length;
      if (!previous && offset.matchIdx + 1 < numPageMatches || previous && offset.matchIdx > 0) {
        offset.matchIdx = previous ? offset.matchIdx - 1 : offset.matchIdx + 1;
        this.#updateMatch(true);
        return;
      }
      this.#advanceOffsetPage(previous);
    }
    this.#nextPageMatch();
  }
  #matchesReady(matches) {
    const offset = this._offset;
    const numMatches = matches.length;
    const previous = this.#state.findPrevious;
    if (numMatches) {
      offset.matchIdx = previous ? numMatches - 1 : 0;
      this.#updateMatch(true);
      return true;
    }
    this.#advanceOffsetPage(previous);
    if (offset.wrapped) {
      offset.matchIdx = null;
      if (this._pagesToSearch < 0) {
        this.#updateMatch(false);
        return true;
      }
    }
    return false;
  }
  #nextPageMatch() {
    if (this._resumePageIdx !== null) {
      console.error("There can only be one pending page.");
    }
    let matches = null;
    do {
      const pageIdx = this._offset.pageIdx;
      matches = this._pageMatches[pageIdx];
      if (!matches) {
        this._resumePageIdx = pageIdx;
        break;
      }
    } while (!this.#matchesReady(matches));
  }
  #advanceOffsetPage(previous) {
    const offset = this._offset;
    const numPages = this._linkService.pagesCount;
    offset.pageIdx = previous ? offset.pageIdx - 1 : offset.pageIdx + 1;
    offset.matchIdx = null;
    this._pagesToSearch--;
    if (offset.pageIdx >= numPages || offset.pageIdx < 0) {
      offset.pageIdx = previous ? numPages - 1 : 0;
      offset.wrapped = true;
    }
  }
  #updateMatch(found = false) {
    let state = FindState.NOT_FOUND;
    const wrapped = this._offset.wrapped;
    this._offset.wrapped = false;
    if (found) {
      const previousPage = this._selected.pageIdx;
      this._selected.pageIdx = this._offset.pageIdx;
      this._selected.matchIdx = this._offset.matchIdx;
      state = wrapped ? FindState.WRAPPED : FindState.FOUND;
      if (previousPage !== -1 && previousPage !== this._selected.pageIdx) {
        this.#updatePage(previousPage);
      }
    }
    this.#updateUIState(state, this.#state.findPrevious);
    if (this._selected.pageIdx !== -1) {
      this._scrollMatches = true;
      this.#updatePage(this._selected.pageIdx);
    }
  }
  #onPagesEdited({
    pagesMapper,
    type,
    pageNumbers
  }) {
    if (this._extractTextPromises.length === 0) {
      return;
    }
    if (type === "copy") {
      const promises = new Map();
      const contents = new Map();
      const diffs = new Map();
      const diacritics = new Map();
      for (const pageNum of pageNumbers) {
        promises.set(pageNum, this._extractTextPromises[pageNum - 1]);
        contents.set(pageNum, this._pageContents[pageNum - 1]);
        diffs.set(pageNum, this._pageDiffs[pageNum - 1]);
        diacritics.set(pageNum, this._hasDiacritics[pageNum - 1]);
      }
      this.#copiedPageData = {
        promises,
        contents,
        diffs,
        diacritics
      };
      return;
    }
    if (type === "cancelCopy") {
      this.#copiedPageData = null;
      return;
    }
    if (type === "delete") {
      this.#savedPageData = {
        promises: this._extractTextPromises,
        contents: this._pageContents,
        diffs: this._pageDiffs,
        diacritics: this._hasDiacritics
      };
    }
    if (type === "cancelDelete") {
      this._extractTextPromises = this.#savedPageData.promises;
      this._pageContents = this.#savedPageData.contents;
      this._pageDiffs = this.#savedPageData.diffs;
      this._hasDiacritics = this.#savedPageData.diacritics;
      return;
    }
    if (type === "cleanSavedData") {
      this.#savedPageData = null;
      return;
    }
    if (this._findTimeout) {
      clearTimeout(this._findTimeout);
      this._findTimeout = null;
    }
    this._resumePageIdx = null;
    this._dirtyMatch = true;
    const prevPromises = this._extractTextPromises;
    const prevContents = this._pageContents;
    const prevDiffs = this._pageDiffs;
    const prevDiacritics = this._hasDiacritics;
    const extractTextPromises = this._extractTextPromises = [];
    const pageContents = this._pageContents = [];
    const pageDiffs = this._pageDiffs = [];
    const hasDiacritics = this._hasDiacritics = [];
    for (let i = 1, ii = pagesMapper.pagesNumber; i <= ii; i++) {
      const prevPageNumber = pagesMapper.getPrevPageNumber(i);
      if (prevPageNumber < 0) {
        const src = -prevPageNumber;
        extractTextPromises.push(this.#copiedPageData?.promises.get(src) || Promise.resolve());
        pageContents.push(this.#copiedPageData?.contents.get(src) ?? "");
        pageDiffs.push(this.#copiedPageData?.diffs.get(src) ?? null);
        hasDiacritics.push(this.#copiedPageData?.diacritics.get(src) ?? false);
        continue;
      }
      extractTextPromises.push(prevPromises[prevPageNumber - 1] || Promise.resolve());
      pageContents.push(prevContents[prevPageNumber - 1] ?? "");
      pageDiffs.push(prevDiffs[prevPageNumber - 1] ?? null);
      hasDiacritics.push(prevDiacritics[prevPageNumber - 1] ?? false);
    }
    if (this.#state) {
      this.#nextMatch();
    }
  }
  #onFindBarClose(evt) {
    const pdfDocument = this._pdfDocument;
    this._firstPageCapability.promise.then(() => {
      if (!this._pdfDocument || pdfDocument && this._pdfDocument !== pdfDocument) {
        return;
      }
      if (this._findTimeout) {
        clearTimeout(this._findTimeout);
        this._findTimeout = null;
      }
      if (this._resumePageIdx) {
        this._resumePageIdx = null;
        this._dirtyMatch = true;
      }
      this.#updateUIState(FindState.FOUND);
      this._highlightMatches = false;
      this.#updateAllPages();
    });
  }
  #requestMatchesCount() {
    const {
      pageIdx,
      matchIdx
    } = this._selected;
    let current = 0,
      total = this._matchesCountTotal;
    if (matchIdx !== -1) {
      for (let i = 0; i < pageIdx; i++) {
        current += this._pageMatches[i]?.length || 0;
      }
      current += matchIdx + 1;
    }
    if (current < 1 || current > total) {
      current = total = 0;
    }
    return {
      current,
      total
    };
  }
  #updateUIResultsCount() {
    this._eventBus.dispatch("updatefindmatchescount", {
      source: this,
      matchesCount: this.#requestMatchesCount()
    });
  }
  #updateUIState(state, previous = false) {
    if (!this.#updateMatchesCountOnProgress && (this.#visitedPagesCount !== this._linkService.pagesCount || state === FindState.PENDING)) {
      return;
    }
    this._eventBus.dispatch("updatefindcontrolstate", {
      source: this,
      state,
      previous,
      entireWord: this.#state?.entireWord ?? null,
      matchesCount: this.#requestMatchesCount(),
      rawQuery: this.#state?.query ?? null
    });
  }
}

// EXTERNAL MODULE: ./node_modules/core-js/modules/es.json.parse.js
var es_json_parse = __webpack_require__(9112);
// EXTERNAL MODULE: ./node_modules/core-js/modules/web.url.parse.js
var web_url_parse = __webpack_require__(5781);
;// ./web/pdf_link_service.js








const DEFAULT_LINK_REL = "noopener noreferrer nofollow";
const LinkTarget = {
  NONE: 0,
  SELF: 1,
  BLANK: 2,
  PARENT: 3,
  TOP: 4
};
class PDFLinkService {
  externalLinkEnabled = true;
  constructor({
    eventBus,
    externalLinkTarget = null,
    externalLinkRel = null,
    ignoreDestinationZoom = false
  } = {}) {
    this.eventBus = eventBus;
    this.externalLinkTarget = externalLinkTarget;
    this.externalLinkRel = externalLinkRel;
    this._ignoreDestinationZoom = ignoreDestinationZoom;
    this.baseUrl = null;
    this.pdfDocument = null;
    this.pdfViewer = null;
    this.pdfHistory = null;
  }
  setDocument(pdfDocument, baseUrl = null) {
    this.baseUrl = baseUrl;
    this.pdfDocument = pdfDocument;
  }
  setViewer(pdfViewer) {
    this.pdfViewer = pdfViewer;
  }
  setHistory(pdfHistory) {
    this.pdfHistory = pdfHistory;
  }
  get pagesCount() {
    return this.pdfDocument?.pagesMapper.pagesNumber || 0;
  }
  get page() {
    return this.pdfDocument ? this.pdfViewer.currentPageNumber : 1;
  }
  set page(value) {
    if (this.pdfDocument) {
      this.pdfViewer.currentPageNumber = value;
    }
  }
  get rotation() {
    return this.pdfDocument ? this.pdfViewer.pagesRotation : 0;
  }
  set rotation(value) {
    if (this.pdfDocument) {
      this.pdfViewer.pagesRotation = value;
    }
  }
  get isInPresentationMode() {
    return this.pdfDocument ? this.pdfViewer.isInPresentationMode : false;
  }
  async goToDestination(dest) {
    if (!this.pdfDocument) {
      return;
    }
    let namedDest, explicitDest, pageNumber;
    if (typeof dest === "string") {
      namedDest = dest;
      explicitDest = await this.pdfDocument.getDestination(dest);
    } else {
      namedDest = null;
      explicitDest = await dest;
    }
    if (!Array.isArray(explicitDest)) {
      console.error(`goToDestination: "${explicitDest}" is not a valid destination array, for dest="${dest}".`);
      return;
    }
    const [destRef] = explicitDest;
    if (destRef && typeof destRef === "object") {
      pageNumber = this.pdfDocument.cachedPageNumber(destRef);
      if (!pageNumber) {
        try {
          pageNumber = (await this.pdfDocument.getPageIndex(destRef)) + 1;
        } catch {
          console.error(`goToDestination: "${destRef}" is not a valid page reference, for dest="${dest}".`);
          return;
        }
      }
    } else if (Number.isInteger(destRef)) {
      pageNumber = destRef + 1;
    }
    if (!pageNumber || pageNumber < 1 || pageNumber > this.pagesCount) {
      console.error(`goToDestination: "${pageNumber}" is not a valid page number, for dest="${dest}".`);
      return;
    }
    if (this.pdfHistory) {
      this.pdfHistory.pushCurrentPosition();
      this.pdfHistory.push({
        namedDest,
        explicitDest,
        pageNumber
      });
    }
    this.pdfViewer.scrollPageIntoView({
      pageNumber,
      destArray: explicitDest,
      ignoreDestinationZoom: this._ignoreDestinationZoom
    });
    const ac = new AbortController();
    this.eventBus.on("textlayerrendered", evt => {
      if (evt.pageNumber === pageNumber) {
        evt.source.textLayer.div.focus();
        ac.abort();
      }
    }, {
      signal: ac.signal,
      ...internalOpt
    });
  }
  goToPage(val) {
    if (!this.pdfDocument) {
      return;
    }
    const pageNumber = typeof val === "string" && this.pdfViewer.pageLabelToPageNumber(val) || val | 0;
    if (!(Number.isInteger(pageNumber) && pageNumber > 0 && pageNumber <= this.pagesCount)) {
      console.error(`PDFLinkService.goToPage: "${val}" is not a valid page.`);
      return;
    }
    if (this.pdfHistory) {
      this.pdfHistory.pushCurrentPosition();
      this.pdfHistory.pushPage(pageNumber);
    }
    this.pdfViewer.scrollPageIntoView({
      pageNumber
    });
  }
  goToXY(pageNumber, x, y, options = {}) {
    this.pdfViewer.scrollPageIntoView({
      pageNumber,
      destArray: [null, {
        name: "XYZ"
      }, x, y],
      ignoreDestinationZoom: true,
      ...options
    });
  }
  async getAttachmentContent(id) {
    try {
      return await this.pdfDocument?.getAttachmentContent(id);
    } catch (error) {
      if (!(error instanceof PasswordException)) {
        console.warn(`Unable to load attachment content: ${error}`);
      }
    }
    return null;
  }
  addLinkAttributes(link, url, newWindow = false) {
    if (!url || typeof url !== "string") {
      throw new Error('A valid "url" parameter must provided.');
    }
    const target = newWindow ? LinkTarget.BLANK : this.externalLinkTarget,
      rel = this.externalLinkRel;
    let displayUrl = url;
    const parsedUrl = URL.parse(url);
    if (parsedUrl?.username || parsedUrl?.password) {
      parsedUrl.username = parsedUrl.password = "";
      displayUrl = parsedUrl.href;
    }
    if (this.externalLinkEnabled) {
      link.href = url;
      link.title = displayUrl;
    } else {
      link.href = "";
      link.title = `Disabled: ${displayUrl}`;
      link.onclick = () => false;
    }
    let targetStr = "";
    switch (target) {
      case LinkTarget.NONE:
        break;
      case LinkTarget.SELF:
        targetStr = "_self";
        break;
      case LinkTarget.BLANK:
        targetStr = "_blank";
        break;
      case LinkTarget.PARENT:
        targetStr = "_parent";
        break;
      case LinkTarget.TOP:
        targetStr = "_top";
        break;
    }
    link.target = targetStr;
    link.rel = typeof rel === "string" ? rel : DEFAULT_LINK_REL;
  }
  getDestinationHash(dest) {
    if (typeof dest === "string") {
      if (dest.length > 0) {
        return this.getAnchorUrl("#" + escape(dest));
      }
    } else if (Array.isArray(dest)) {
      const str = JSON.stringify(dest);
      if (str.length > 0) {
        return this.getAnchorUrl("#" + escape(str));
      }
    }
    return this.getAnchorUrl("");
  }
  getAnchorUrl(anchor) {
    return this.baseUrl ? this.baseUrl + anchor : anchor;
  }
  setHash(hash) {
    if (!this.pdfDocument) {
      return;
    }
    let pageNumber, dest;
    if (hash.includes("=")) {
      const params = parseQueryString(hash);
      if (params.has("search")) {
        const query = params.get("search").replaceAll('"', ""),
          phrase = params.get("phrase") === "true";
        this.eventBus.dispatch("findfromurlhash", {
          source: this,
          query: phrase ? query : query.match(/\S+/g)
        });
      }
      if (params.has("page")) {
        pageNumber = params.get("page") | 0 || 1;
      }
      if (params.has("zoom")) {
        const zoomArgs = params.get("zoom").split(",");
        const zoomArg = zoomArgs[0];
        const zoomArgNumber = parseFloat(zoomArg);
        if (!zoomArg.includes("Fit")) {
          dest = [null, {
            name: "XYZ"
          }, zoomArgs.length > 1 ? zoomArgs[1] | 0 : null, zoomArgs.length > 2 ? zoomArgs[2] | 0 : null, zoomArgNumber ? zoomArgNumber / 100 : zoomArg];
        } else if (zoomArg === "Fit" || zoomArg === "FitB") {
          dest = [null, {
            name: zoomArg
          }];
        } else if (zoomArg === "FitH" || zoomArg === "FitBH" || zoomArg === "FitV" || zoomArg === "FitBV") {
          dest = [null, {
            name: zoomArg
          }, zoomArgs.length > 1 ? zoomArgs[1] | 0 : null];
        } else if (zoomArg === "FitR") {
          if (zoomArgs.length !== 5) {
            console.error('PDFLinkService.setHash: Not enough parameters for "FitR".');
          } else {
            dest = [null, {
              name: zoomArg
            }, zoomArgs[1] | 0, zoomArgs[2] | 0, zoomArgs[3] | 0, zoomArgs[4] | 0];
          }
        } else {
          console.error(`PDFLinkService.setHash: "${zoomArg}" is not a valid zoom value.`);
        }
      }
      if (dest) {
        this.pdfViewer.scrollPageIntoView({
          pageNumber: pageNumber || this.page,
          destArray: dest,
          allowNegativeOffset: true
        });
      } else if (pageNumber) {
        this.page = pageNumber;
      }
      if (params.has("pagemode")) {
        this.eventBus.dispatch("pagemode", {
          source: this,
          mode: params.get("pagemode")
        });
      }
      if (params.has("nameddest")) {
        this.goToDestination(params.get("nameddest"));
      }
      return;
    }
    dest = unescape(hash);
    try {
      dest = JSON.parse(dest);
      if (!Array.isArray(dest)) {
        dest = dest.toString();
      }
    } catch {}
    if (typeof dest === "string" || isValidExplicitDest(dest)) {
      this.goToDestination(dest);
      return;
    }
    console.error(`PDFLinkService.setHash: "${unescape(hash)}" is not a valid destination.`);
  }
  executeNamedAction(action) {
    if (!this.pdfDocument) {
      return;
    }
    switch (action) {
      case "GoBack":
        this.pdfHistory?.back();
        break;
      case "GoForward":
        this.pdfHistory?.forward();
        break;
      case "NextPage":
        this.pdfViewer.nextPage();
        break;
      case "PrevPage":
        this.pdfViewer.previousPage();
        break;
      case "LastPage":
        this.page = this.pagesCount;
        break;
      case "FirstPage":
        this.page = 1;
        break;
      default:
        break;
    }
    this.eventBus.dispatch("namedaction", {
      source: this,
      action
    });
  }
  async executeSetOCGState(action) {
    if (!this.pdfDocument) {
      return;
    }
    const pdfDocument = this.pdfDocument,
      optionalContentConfig = await this.pdfViewer.optionalContentConfigPromise;
    if (pdfDocument !== this.pdfDocument) {
      return;
    }
    optionalContentConfig.setOCGState(action);
    this.pdfViewer.optionalContentConfigPromise = Promise.resolve(optionalContentConfig);
  }
}
class SimpleLinkService extends PDFLinkService {
  setDocument(pdfDocument, baseUrl = null) {}
}

;// ./web/annotation_layer_builder.js






class AnnotationLayerBuilder {
  #annotations = null;
  #commentManager = null;
  #onAppend = null;
  #eventAC = null;
  #linksInjected = false;
  constructor({
    pdfPage,
    linkService,
    downloadManager,
    annotationStorage = null,
    imageResourcesPath = "",
    renderForms = true,
    enableComment = false,
    commentManager = null,
    enableScripting = false,
    hasJSActionsPromise = null,
    fieldObjectsPromise = null,
    annotationCanvasMap = null,
    accessibilityManager = null,
    annotationEditorUIManager = null,
    onAppend = null
  }) {
    this.pdfPage = pdfPage;
    this.linkService = linkService;
    this.downloadManager = downloadManager;
    this.imageResourcesPath = imageResourcesPath;
    this.renderForms = renderForms;
    this.annotationStorage = annotationStorage;
    this.enableComment = enableComment;
    this.#commentManager = commentManager;
    this.enableScripting = enableScripting;
    this._hasJSActionsPromise = hasJSActionsPromise || Promise.resolve(false);
    this._fieldObjectsPromise = fieldObjectsPromise || Promise.resolve(null);
    this._annotationCanvasMap = annotationCanvasMap;
    this._accessibilityManager = accessibilityManager;
    this._annotationEditorUIManager = annotationEditorUIManager;
    this.#onAppend = onAppend;
    this.annotationLayer = null;
    this.div = null;
    this._cancelled = false;
    this._eventBus = linkService.eventBus;
  }
  async render({
    viewport,
    intent = "display",
    structTreeLayer = null,
    optionalContentConfigPromise = null
  }) {
    if (this.div) {
      const optionalContentConfig = await optionalContentConfigPromise;
      if (this._cancelled || !this.annotationLayer) {
        return;
      }
      this.annotationLayer.update({
        viewport: viewport.clone({
          dontFlip: true
        }),
        optionalContentConfig
      });
      return;
    }
    const [annotations, hasJSActions, fieldObjects, optionalContentConfig] = await Promise.all([this.pdfPage.getAnnotations({
      intent
    }), this._hasJSActionsPromise, this._fieldObjectsPromise, optionalContentConfigPromise]);
    if (this._cancelled) {
      return;
    }
    const div = this.div = document.createElement("div");
    div.className = "annotationLayer";
    this.#onAppend?.(div);
    this.#initAnnotationLayer(viewport, structTreeLayer);
    if (annotations.length === 0) {
      this.#annotations = annotations;
      setLayerDimensions(this.div, viewport);
      return;
    }
    await this.annotationLayer.render({
      annotations,
      imageResourcesPath: this.imageResourcesPath,
      renderForms: this.renderForms,
      downloadManager: this.downloadManager,
      enableComment: this.enableComment,
      enableScripting: this.enableScripting,
      hasJSActions,
      fieldObjects,
      optionalContentConfig
    });
    this.#annotations = annotations;
    if (this.linkService.isInPresentationMode) {
      this.#updatePresentationModeState(PresentationModeState.FULLSCREEN);
    }
    if (!this.#eventAC) {
      this.#eventAC = new AbortController();
      this._eventBus?.on("presentationmodechanged", evt => {
        this.#updatePresentationModeState(evt.state);
      }, {
        signal: this.#eventAC.signal,
        ...internalOpt
      });
    }
  }
  #initAnnotationLayer(viewport, structTreeLayer) {
    this.annotationLayer = new AnnotationLayer({
      div: this.div,
      accessibilityManager: this._accessibilityManager,
      annotationCanvasMap: this._annotationCanvasMap,
      annotationEditorUIManager: this._annotationEditorUIManager,
      annotationStorage: this.annotationStorage,
      page: this.pdfPage,
      viewport: viewport.clone({
        dontFlip: true
      }),
      structTreeLayer,
      commentManager: this.#commentManager,
      linkService: this.linkService
    });
  }
  cancel() {
    this._cancelled = true;
    this.#eventAC?.abort();
    this.#eventAC = null;
    this.annotationLayer?.destroy();
  }
  refreshCanvases() {
    this.annotationLayer?.refreshCanvases();
  }
  hide() {
    if (!this.div) {
      return;
    }
    this.div.hidden = true;
  }
  hasEditableAnnotations() {
    return !!this.annotationLayer?.hasEditableAnnotations();
  }
  async injectLinkAnnotations(inferredLinks) {
    if (this.#annotations === null) {
      throw new Error("`render` method must be called before `injectLinkAnnotations`.");
    }
    if (this._cancelled || this.#linksInjected) {
      return;
    }
    this.#linksInjected = true;
    const newLinks = this.#annotations.length ? this.#checkInferredLinks(inferredLinks) : inferredLinks;
    if (!newLinks.length) {
      return;
    }
    await this.annotationLayer.addLinkAnnotations(newLinks);
  }
  #updatePresentationModeState(state) {
    if (!this.div) {
      return;
    }
    let disableFormElements = false;
    switch (state) {
      case PresentationModeState.FULLSCREEN:
        disableFormElements = true;
        break;
      case PresentationModeState.NORMAL:
        break;
      default:
        return;
    }
    for (const section of this.div.childNodes) {
      if (section.hasAttribute("data-internal-link") || section.classList.contains("mediaAnnotation")) {
        continue;
      }
      section.inert = disableFormElements;
    }
  }
  #checkInferredLinks(inferredLinks) {
    function annotationRects(annot) {
      if (!annot.quadPoints) {
        return [annot.rect];
      }
      const rects = [];
      for (let i = 2, ii = annot.quadPoints.length; i < ii; i += 8) {
        const trX = annot.quadPoints[i];
        const trY = annot.quadPoints[i + 1];
        const blX = annot.quadPoints[i + 2];
        const blY = annot.quadPoints[i + 3];
        rects.push([blX, blY, trX, trY]);
      }
      return rects;
    }
    function intersectAnnotations(annot1, annot2) {
      const intersections = [];
      const annot1Rects = annotationRects(annot1);
      const annot2Rects = annotationRects(annot2);
      for (const rect1 of annot1Rects) {
        for (const rect2 of annot2Rects) {
          const intersection = Util.intersect(rect1, rect2);
          if (intersection) {
            intersections.push(intersection);
          }
        }
      }
      return intersections;
    }
    function areaRects(rects) {
      let totalArea = 0;
      for (const rect of rects) {
        totalArea += Math.abs((rect[2] - rect[0]) * (rect[3] - rect[1]));
      }
      return totalArea;
    }
    return inferredLinks.filter(link => {
      let linkAreaRects;
      for (const annotation of this.#annotations) {
        if (annotation.annotationType !== AnnotationType.LINK) {
          continue;
        }
        const intersections = intersectAnnotations(annotation, link);
        if (intersections.length === 0) {
          continue;
        }
        linkAreaRects ??= areaRects(annotationRects(link));
        if (areaRects(intersections) / linkAreaRects > 0.5) {
          return false;
        }
      }
      return true;
    });
  }
}

// EXTERNAL MODULE: ./node_modules/core-js/modules/es.weak-map.get-or-insert.js
var es_weak_map_get_or_insert = __webpack_require__(8454);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.weak-map.get-or-insert-computed.js
var es_weak_map_get_or_insert_computed = __webpack_require__(9452);
;// ./web/base_download_manager.js



class BaseDownloadManager {
  #openBlobUrls = new WeakMap();
  _triggerDownload(blobUrl, originalUrl, filename, isAttachment = false) {
    throw new Error("Not implemented: _triggerDownload");
  }
  _getOpenDataUrl(blobUrl, filename, dest = null) {
    throw new Error("Not implemented: _getOpenDataUrl");
  }
  downloadData(data, filename, contentType) {
    const blobUrl = URL.createObjectURL(new Blob([data], {
      type: contentType
    }));
    this._triggerDownload(blobUrl, blobUrl, filename, true);
  }
  openOrDownloadData(data, filename, dest = null) {
    const isPdfData = isPdfFile(filename);
    const contentType = isPdfData ? "application/pdf" : "";
    if (isPdfData) {
      const blobUrl = this.#openBlobUrls.getOrInsertComputed(data, () => URL.createObjectURL(new Blob([data], {
        type: contentType
      })));
      try {
        const viewerUrl = this._getOpenDataUrl(blobUrl, filename, dest);
        window.open(viewerUrl);
        return true;
      } catch (ex) {
        console.error("openOrDownloadData:", ex);
        URL.revokeObjectURL(blobUrl);
        this.#openBlobUrls.delete(data);
      }
    }
    this.downloadData(data, filename, contentType);
    return false;
  }
  download(data, url, filename) {
    const blobUrl = data ? URL.createObjectURL(new Blob([data], {
      type: "application/pdf"
    })) : null;
    this._triggerDownload(blobUrl, url, filename);
  }
}

;// ./web/download_manager.js


class DownloadManager extends BaseDownloadManager {
  _triggerDownload(blobUrl, originalUrl, filename, isAttachment = false) {
    if (!blobUrl && !isAttachment) {
      if (!createValidAbsoluteUrl(originalUrl, "http://example.com")) {
        throw new Error(`_triggerDownload - not a valid URL: ${originalUrl}`);
      }
      blobUrl = originalUrl + "#pdfjs.action=download";
    }
    const a = document.createElement("a");
    a.href = blobUrl;
    a.target = "_parent";
    if ("download" in a) {
      a.download = filename;
    }
    (document.body || document.documentElement).append(a);
    a.click();
    a.remove();
  }
  _getOpenDataUrl(blobUrl, filename, dest = null) {
    throw new Error("Opening data is not supported in `COMPONENTS` builds.");
  }
}

// EXTERNAL MODULE: ./node_modules/core-js/modules/es.iterator.find.js
var es_iterator_find = __webpack_require__(116);
;// ./web/event_utils.js














const WaitOnType = {
  EVENT: "event",
  TIMEOUT: "timeout"
};
async function waitOnEventOrTimeout({
  target,
  name,
  delay = 0
}) {
  if (typeof target !== "object" || !(name && typeof name === "string") || !(Number.isInteger(delay) && delay >= 0)) {
    throw new Error("waitOnEventOrTimeout - invalid parameters.");
  }
  const {
    promise,
    resolve
  } = Promise.withResolvers();
  const ac = new AbortController();
  function handler(type) {
    ac.abort();
    clearTimeout(timeout);
    resolve(type);
  }
  const evtMethod = target instanceof EventBus ? "on" : "addEventListener";
  const evtOpts = target instanceof EventBus ? {
    signal: ac.signal,
    ...internalOpt
  } : {
    signal: ac.signal
  };
  target[evtMethod](name, handler.bind(null, WaitOnType.EVENT), evtOpts);
  const timeout = setTimeout(handler.bind(null, WaitOnType.TIMEOUT), delay);
  return promise;
}
class EventBus {
  #listeners = new Map();
  constructor() {
    Object.seal(this);
  }
  on(eventName, listener, options = null) {
    let rmAbort = null;
    if (options?.signal instanceof AbortSignal) {
      const {
        signal
      } = options;
      if (signal.aborted) {
        console.error("Cannot use an `aborted` signal.");
        return;
      }
      const onAbort = () => this.off(eventName, listener);
      rmAbort = () => signal.removeEventListener("abort", onAbort);
      signal.addEventListener("abort", onAbort);
    }
    this.#listeners.getOrInsertComputed(eventName, makeSet).add({
      listener,
      internal: options?.internal === INTERNAL_EVT,
      once: options?.once === true,
      rmAbort
    });
  }
  off(eventName, listener, options = null) {
    const eventListeners = this.#listeners.get(eventName);
    const evt = eventListeners?.keys().find(e => e.listener === listener);
    if (evt) {
      evt.rmAbort?.();
      eventListeners.delete(evt);
    }
  }
  dispatch(eventName, data) {
    const eventListeners = this.#listeners.get(eventName);
    if (!eventListeners?.size) {
      return;
    }
    let extListeners;
    for (const {
      listener,
      internal,
      once
    } of new Set(eventListeners)) {
      if (once) {
        this.off(eventName, listener);
      }
      if (!internal) {
        (extListeners ??= []).push(listener);
        continue;
      }
      listener(data);
    }
    if (extListeners) {
      for (const listener of extListeners) {
        listener(data);
      }
    }
  }
}
class FirefoxEventBus extends EventBus {
  #externalServices;
  #globalEventNames;
  #isInAutomation;
  constructor(globalEventNames, externalServices, isInAutomation) {
    super();
    this.#globalEventNames = globalEventNames;
    this.#externalServices = externalServices;
    this.#isInAutomation = isInAutomation;
  }
  dispatch(eventName, data) {
    throw new Error("Not implemented: FirefoxEventBus.dispatch");
  }
}

;// ./node_modules/@fluent/bundle/esm/types.js
class FluentType {
  constructor(value) {
    this.value = value;
  }
  valueOf() {
    return this.value;
  }
}
class FluentNone extends FluentType {
  constructor(value = "???") {
    super(value);
  }
  toString(scope) {
    return `{${this.value}}`;
  }
}
class FluentNumber extends FluentType {
  constructor(value, opts = {}) {
    super(value);
    this.opts = opts;
  }
  toString(scope) {
    if (scope) {
      try {
        const nf = scope.memoizeIntlObject(Intl.NumberFormat, this.opts);
        return nf.format(this.value);
      } catch (err) {
        scope.reportError(err);
      }
    }
    return this.value.toString(10);
  }
}
class FluentDateTime extends FluentType {
  static supportsValue(value) {
    if (typeof value === "number") return true;
    if (value instanceof Date) return true;
    if (value instanceof FluentType) return FluentDateTime.supportsValue(value.valueOf());
    if ("Temporal" in globalThis) {
      const _Temporal = globalThis.Temporal;
      if (value instanceof _Temporal.Instant || value instanceof _Temporal.PlainDateTime || value instanceof _Temporal.PlainDate || value instanceof _Temporal.PlainMonthDay || value instanceof _Temporal.PlainTime || value instanceof _Temporal.PlainYearMonth) {
        return true;
      }
    }
    return false;
  }
  constructor(value, opts = {}) {
    if (value instanceof FluentDateTime) {
      opts = {
        ...value.opts,
        ...opts
      };
      value = value.value;
    } else if (value instanceof FluentType) {
      value = value.valueOf();
    }
    if (typeof value === "object" && "calendarId" in value && opts.calendar === undefined) {
      opts = {
        ...opts,
        calendar: value.calendarId
      };
    }
    super(value);
    this.opts = opts;
  }
  [Symbol.toPrimitive](hint) {
    return hint === "string" ? this.toString() : this.toNumber();
  }
  toNumber() {
    const value = this.value;
    if (typeof value === "number") return value;
    if (value instanceof Date) return value.getTime();
    if ("epochMilliseconds" in value) {
      return value.epochMilliseconds;
    }
    if ("toZonedDateTime" in value) {
      return value.toZonedDateTime("UTC").epochMilliseconds;
    }
    throw new TypeError("Unwrapping a non-number value as a number");
  }
  toString(scope) {
    if (scope) {
      try {
        const dtf = scope.memoizeIntlObject(Intl.DateTimeFormat, this.opts);
        return dtf.format(this.value);
      } catch (err) {
        scope.reportError(err);
      }
    }
    if (typeof this.value === "number" || this.value instanceof Date) {
      return new Date(this.value).toISOString();
    }
    return this.value.toString();
  }
}
;// ./node_modules/@fluent/bundle/esm/resolver.js


const MAX_PLACEABLES = 100;
const FSI = "\u2068";
const PDI = "\u2069";
function match(scope, selector, key) {
  if (key === selector) {
    return true;
  }
  if (key instanceof FluentNumber && selector instanceof FluentNumber && key.value === selector.value) {
    return true;
  }
  if (selector instanceof FluentNumber && typeof key === "string") {
    let category = scope.memoizeIntlObject(Intl.PluralRules, selector.opts).select(selector.value);
    if (key === category) {
      return true;
    }
  }
  return false;
}
function getDefault(scope, variants, star) {
  if (variants[star]) {
    return resolvePattern(scope, variants[star].value);
  }
  scope.reportError(new RangeError("No default"));
  return new FluentNone();
}
function getArguments(scope, args) {
  const positional = [];
  const named = Object.create(null);
  for (const arg of args) {
    if (arg.type === "narg") {
      named[arg.name] = resolveExpression(scope, arg.value);
    } else {
      positional.push(resolveExpression(scope, arg));
    }
  }
  return {
    positional,
    named
  };
}
function resolveExpression(scope, expr) {
  switch (expr.type) {
    case "str":
      return expr.value;
    case "num":
      return new FluentNumber(expr.value, {
        minimumFractionDigits: expr.precision
      });
    case "var":
      return resolveVariableReference(scope, expr);
    case "mesg":
      return resolveMessageReference(scope, expr);
    case "term":
      return resolveTermReference(scope, expr);
    case "func":
      return resolveFunctionReference(scope, expr);
    case "select":
      return resolveSelectExpression(scope, expr);
    default:
      return new FluentNone();
  }
}
function resolveVariableReference(scope, {
  name
}) {
  let arg;
  if (scope.params) {
    if (Object.prototype.hasOwnProperty.call(scope.params, name)) {
      arg = scope.params[name];
    } else {
      return new FluentNone(`$${name}`);
    }
  } else if (scope.args && Object.prototype.hasOwnProperty.call(scope.args, name)) {
    arg = scope.args[name];
  } else {
    scope.reportError(new ReferenceError(`Unknown variable: $${name}`));
    return new FluentNone(`$${name}`);
  }
  if (arg instanceof FluentType) {
    return arg;
  }
  switch (typeof arg) {
    case "string":
      return arg;
    case "number":
      return new FluentNumber(arg);
    case "object":
      if (FluentDateTime.supportsValue(arg)) {
        return new FluentDateTime(arg);
      }
    default:
      scope.reportError(new TypeError(`Variable type not supported: $${name}, ${typeof arg}`));
      return new FluentNone(`$${name}`);
  }
}
function resolveMessageReference(scope, {
  name,
  attr
}) {
  const message = scope.bundle._messages.get(name);
  if (!message) {
    scope.reportError(new ReferenceError(`Unknown message: ${name}`));
    return new FluentNone(name);
  }
  if (attr) {
    const attribute = message.attributes[attr];
    if (attribute) {
      return resolvePattern(scope, attribute);
    }
    scope.reportError(new ReferenceError(`Unknown attribute: ${attr}`));
    return new FluentNone(`${name}.${attr}`);
  }
  if (message.value) {
    return resolvePattern(scope, message.value);
  }
  scope.reportError(new ReferenceError(`No value: ${name}`));
  return new FluentNone(name);
}
function resolveTermReference(scope, {
  name,
  attr,
  args
}) {
  const id = `-${name}`;
  const term = scope.bundle._terms.get(id);
  if (!term) {
    scope.reportError(new ReferenceError(`Unknown term: ${id}`));
    return new FluentNone(id);
  }
  if (attr) {
    const attribute = term.attributes[attr];
    if (attribute) {
      scope.params = getArguments(scope, args).named;
      const resolved = resolvePattern(scope, attribute);
      scope.params = null;
      return resolved;
    }
    scope.reportError(new ReferenceError(`Unknown attribute: ${attr}`));
    return new FluentNone(`${id}.${attr}`);
  }
  scope.params = getArguments(scope, args).named;
  const resolved = resolvePattern(scope, term.value);
  scope.params = null;
  return resolved;
}
function resolveFunctionReference(scope, {
  name,
  args
}) {
  let func = scope.bundle._functions[name];
  if (!func) {
    scope.reportError(new ReferenceError(`Unknown function: ${name}()`));
    return new FluentNone(`${name}()`);
  }
  if (typeof func !== "function") {
    scope.reportError(new TypeError(`Function ${name}() is not callable`));
    return new FluentNone(`${name}()`);
  }
  try {
    let resolved = getArguments(scope, args);
    return func(resolved.positional, resolved.named);
  } catch (err) {
    scope.reportError(err);
    return new FluentNone(`${name}()`);
  }
}
function resolveSelectExpression(scope, {
  selector,
  variants,
  star
}) {
  let sel = resolveExpression(scope, selector);
  if (sel instanceof FluentNone) {
    return getDefault(scope, variants, star);
  }
  for (const variant of variants) {
    const key = resolveExpression(scope, variant.key);
    if (match(scope, sel, key)) {
      return resolvePattern(scope, variant.value);
    }
  }
  return getDefault(scope, variants, star);
}
function resolveComplexPattern(scope, ptn) {
  if (scope.dirty.has(ptn)) {
    scope.reportError(new RangeError("Cyclic reference"));
    return new FluentNone();
  }
  scope.dirty.add(ptn);
  const result = [];
  const useIsolating = scope.bundle._useIsolating && ptn.length > 1;
  for (const elem of ptn) {
    if (typeof elem === "string") {
      result.push(scope.bundle._transform(elem));
      continue;
    }
    scope.placeables++;
    if (scope.placeables > MAX_PLACEABLES) {
      scope.dirty.delete(ptn);
      throw new RangeError(`Too many placeables expanded: ${scope.placeables}, ` + `max allowed is ${MAX_PLACEABLES}`);
    }
    if (useIsolating) {
      result.push(FSI);
    }
    result.push(resolveExpression(scope, elem).toString(scope));
    if (useIsolating) {
      result.push(PDI);
    }
  }
  scope.dirty.delete(ptn);
  return result.join("");
}
function resolvePattern(scope, value) {
  if (typeof value === "string") {
    return scope.bundle._transform(value);
  }
  return resolveComplexPattern(scope, value);
}
;// ./node_modules/@fluent/bundle/esm/scope.js


class Scope {
  constructor(bundle, errors, args) {
    this.dirty = new WeakSet();
    this.params = null;
    this.placeables = 0;
    this.bundle = bundle;
    this.errors = errors;
    this.args = args;
  }
  reportError(error) {
    if (!this.errors || !(error instanceof Error)) {
      throw error;
    }
    this.errors.push(error);
  }
  memoizeIntlObject(ctor, opts) {
    let cache = this.bundle._intls.get(ctor);
    if (!cache) {
      cache = {};
      this.bundle._intls.set(ctor, cache);
    }
    let id = JSON.stringify(opts);
    if (!cache[id]) {
      cache[id] = new ctor(this.bundle.locales, opts);
    }
    return cache[id];
  }
}
;// ./node_modules/@fluent/bundle/esm/builtins.js


function values(opts, allowed) {
  const unwrapped = Object.create(null);
  for (const [name, opt] of Object.entries(opts)) {
    if (allowed.includes(name)) {
      unwrapped[name] = opt.valueOf();
    }
  }
  return unwrapped;
}
const NUMBER_ALLOWED = ["unitDisplay", "currencyDisplay", "useGrouping", "minimumIntegerDigits", "minimumFractionDigits", "maximumFractionDigits", "minimumSignificantDigits", "maximumSignificantDigits"];
function NUMBER(args, opts) {
  let arg = args[0];
  if (arg instanceof FluentNone) {
    return new FluentNone(`NUMBER(${arg.valueOf()})`);
  }
  if (arg instanceof FluentNumber) {
    return new FluentNumber(arg.valueOf(), {
      ...arg.opts,
      ...values(opts, NUMBER_ALLOWED)
    });
  }
  if (arg instanceof FluentDateTime) {
    return new FluentNumber(arg.toNumber(), {
      ...values(opts, NUMBER_ALLOWED)
    });
  }
  throw new TypeError("Invalid argument to NUMBER");
}
const DATETIME_ALLOWED = ["dateStyle", "timeStyle", "fractionalSecondDigits", "dayPeriod", "hour12", "weekday", "era", "year", "month", "day", "hour", "minute", "second", "timeZoneName"];
function DATETIME(args, opts) {
  let arg = args[0];
  if (arg instanceof FluentNone) {
    return new FluentNone(`DATETIME(${arg.valueOf()})`);
  }
  if (arg instanceof FluentDateTime || arg instanceof FluentNumber) {
    return new FluentDateTime(arg, values(opts, DATETIME_ALLOWED));
  }
  throw new TypeError("Invalid argument to DATETIME");
}
;// ./node_modules/@fluent/bundle/esm/memoizer.js


const cache = new Map();
function getMemoizerForLocale(locales) {
  const stringLocale = Array.isArray(locales) ? locales.join(" ") : locales;
  let memoizer = cache.get(stringLocale);
  if (memoizer === undefined) {
    memoizer = new Map();
    cache.set(stringLocale, memoizer);
  }
  return memoizer;
}
;// ./node_modules/@fluent/bundle/esm/bundle.js








class FluentBundle {
  constructor(locales, {
    functions,
    useIsolating = true,
    transform = v => v
  } = {}) {
    this._terms = new Map();
    this._messages = new Map();
    this.locales = Array.isArray(locales) ? locales : [locales];
    this._functions = {
      NUMBER: NUMBER,
      DATETIME: DATETIME,
      ...functions
    };
    this._useIsolating = useIsolating;
    this._transform = transform;
    this._intls = getMemoizerForLocale(locales);
  }
  hasMessage(id) {
    return this._messages.has(id);
  }
  getMessage(id) {
    return this._messages.get(id);
  }
  addResource(res, {
    allowOverrides = false
  } = {}) {
    const errors = [];
    for (let i = 0; i < res.body.length; i++) {
      let entry = res.body[i];
      if (entry.id.startsWith("-")) {
        if (allowOverrides === false && this._terms.has(entry.id)) {
          errors.push(new Error(`Attempt to override an existing term: "${entry.id}"`));
          continue;
        }
        this._terms.set(entry.id, entry);
      } else {
        if (allowOverrides === false && this._messages.has(entry.id)) {
          errors.push(new Error(`Attempt to override an existing message: "${entry.id}"`));
          continue;
        }
        this._messages.set(entry.id, entry);
      }
    }
    return errors;
  }
  formatPattern(pattern, args = null, errors = null) {
    if (typeof pattern === "string") {
      return this._transform(pattern);
    }
    let scope = new Scope(this, errors, args);
    try {
      let value = resolveComplexPattern(scope, pattern);
      return value.toString(scope);
    } catch (err) {
      if (scope.errors && err instanceof Error) {
        scope.errors.push(err);
        return new FluentNone().toString(scope);
      }
      throw err;
    }
  }
}
;// ./node_modules/@fluent/bundle/esm/resource.js

const RE_MESSAGE_START = /^(-?[a-zA-Z][\w-]*) *= */gm;
const RE_ATTRIBUTE_START = /\.([a-zA-Z][\w-]*) *= */y;
const RE_VARIANT_START = /\*?\[/y;
const RE_NUMBER_LITERAL = /(-?[0-9]+(?:\.([0-9]+))?)/y;
const RE_IDENTIFIER = /([a-zA-Z][\w-]*)/y;
const RE_REFERENCE = /([$-])?([a-zA-Z][\w-]*)(?:\.([a-zA-Z][\w-]*))?/y;
const RE_FUNCTION_NAME = /^[A-Z][A-Z0-9_-]*$/;
const RE_TEXT_RUN = /([^{}\n\r]+)/y;
const RE_STRING_RUN = /([^\\"\n\r]*)/y;
const RE_STRING_ESCAPE = /\\([\\"])/y;
const RE_UNICODE_ESCAPE = /\\u([a-fA-F0-9]{4})|\\U([a-fA-F0-9]{6})/y;
const RE_LEADING_NEWLINES = /^\n+/;
const RE_TRAILING_SPACES = / +$/;
const RE_BLANK_LINES = / *\r?\n/g;
const RE_INDENT = /( *)$/;
const TOKEN_BRACE_OPEN = /{\s*/y;
const TOKEN_BRACE_CLOSE = /\s*}/y;
const TOKEN_BRACKET_OPEN = /\[\s*/y;
const TOKEN_BRACKET_CLOSE = /\s*] */y;
const TOKEN_PAREN_OPEN = /\s*\(\s*/y;
const TOKEN_ARROW = /\s*->\s*/y;
const TOKEN_COLON = /\s*:\s*/y;
const TOKEN_COMMA = /\s*,?\s*/y;
const TOKEN_BLANK = /\s+/y;
class FluentResource {
  constructor(source) {
    this.body = [];
    RE_MESSAGE_START.lastIndex = 0;
    let cursor = 0;
    while (true) {
      let next = RE_MESSAGE_START.exec(source);
      if (next === null) {
        break;
      }
      cursor = RE_MESSAGE_START.lastIndex;
      try {
        this.body.push(parseMessage(next[1]));
      } catch (err) {
        if (err instanceof SyntaxError) {
          continue;
        }
        throw err;
      }
    }
    function test(re) {
      re.lastIndex = cursor;
      return re.test(source);
    }
    function consumeChar(char, errorClass) {
      if (source[cursor] === char) {
        cursor++;
        return true;
      }
      if (errorClass) {
        throw new errorClass(`Expected ${char}`);
      }
      return false;
    }
    function consumeToken(re, errorClass) {
      if (test(re)) {
        cursor = re.lastIndex;
        return true;
      }
      if (errorClass) {
        throw new errorClass(`Expected ${re.toString()}`);
      }
      return false;
    }
    function match(re) {
      re.lastIndex = cursor;
      let result = re.exec(source);
      if (result === null) {
        throw new SyntaxError(`Expected ${re.toString()}`);
      }
      cursor = re.lastIndex;
      return result;
    }
    function match1(re) {
      return match(re)[1];
    }
    function parseMessage(id) {
      let value = parsePattern();
      let attributes = parseAttributes();
      if (value === null && Object.keys(attributes).length === 0) {
        throw new SyntaxError("Expected message value or attributes");
      }
      return {
        id,
        value,
        attributes
      };
    }
    function parseAttributes() {
      let attrs = Object.create(null);
      while (test(RE_ATTRIBUTE_START)) {
        let name = match1(RE_ATTRIBUTE_START);
        let value = parsePattern();
        if (value === null) {
          throw new SyntaxError("Expected attribute value");
        }
        attrs[name] = value;
      }
      return attrs;
    }
    function parsePattern() {
      let first;
      if (test(RE_TEXT_RUN)) {
        first = match1(RE_TEXT_RUN);
      }
      if (source[cursor] === "{" || source[cursor] === "}") {
        return parsePatternElements(first ? [first] : [], Infinity);
      }
      let indent = parseIndent();
      if (indent) {
        if (first) {
          return parsePatternElements([first, indent], indent.length);
        }
        indent.value = trim(indent.value, RE_LEADING_NEWLINES);
        return parsePatternElements([indent], indent.length);
      }
      if (first) {
        return trim(first, RE_TRAILING_SPACES);
      }
      return null;
    }
    function parsePatternElements(elements = [], commonIndent) {
      while (true) {
        if (test(RE_TEXT_RUN)) {
          elements.push(match1(RE_TEXT_RUN));
          continue;
        }
        if (source[cursor] === "{") {
          elements.push(parsePlaceable());
          continue;
        }
        if (source[cursor] === "}") {
          throw new SyntaxError("Unbalanced closing brace");
        }
        let indent = parseIndent();
        if (indent) {
          elements.push(indent);
          commonIndent = Math.min(commonIndent, indent.length);
          continue;
        }
        break;
      }
      let lastIndex = elements.length - 1;
      let lastElement = elements[lastIndex];
      if (typeof lastElement === "string") {
        elements[lastIndex] = trim(lastElement, RE_TRAILING_SPACES);
      }
      let baked = [];
      for (let element of elements) {
        if (element instanceof Indent) {
          element = element.value.slice(0, element.value.length - commonIndent);
        }
        if (element) {
          baked.push(element);
        }
      }
      return baked;
    }
    function parsePlaceable() {
      consumeToken(TOKEN_BRACE_OPEN, SyntaxError);
      let selector = parseInlineExpression();
      if (consumeToken(TOKEN_BRACE_CLOSE)) {
        return selector;
      }
      if (consumeToken(TOKEN_ARROW)) {
        let variants = parseVariants();
        consumeToken(TOKEN_BRACE_CLOSE, SyntaxError);
        return {
          type: "select",
          selector,
          ...variants
        };
      }
      throw new SyntaxError("Unclosed placeable");
    }
    function parseInlineExpression() {
      if (source[cursor] === "{") {
        return parsePlaceable();
      }
      if (test(RE_REFERENCE)) {
        let [, sigil, name, attr = null] = match(RE_REFERENCE);
        if (sigil === "$") {
          return {
            type: "var",
            name
          };
        }
        if (consumeToken(TOKEN_PAREN_OPEN)) {
          let args = parseArguments();
          if (sigil === "-") {
            return {
              type: "term",
              name,
              attr,
              args
            };
          }
          if (RE_FUNCTION_NAME.test(name)) {
            return {
              type: "func",
              name,
              args
            };
          }
          throw new SyntaxError("Function names must be all upper-case");
        }
        if (sigil === "-") {
          return {
            type: "term",
            name,
            attr,
            args: []
          };
        }
        return {
          type: "mesg",
          name,
          attr
        };
      }
      return parseLiteral();
    }
    function parseArguments() {
      let args = [];
      while (true) {
        switch (source[cursor]) {
          case ")":
            cursor++;
            return args;
          case undefined:
            throw new SyntaxError("Unclosed argument list");
        }
        args.push(parseArgument());
        consumeToken(TOKEN_COMMA);
      }
    }
    function parseArgument() {
      let expr = parseInlineExpression();
      if (expr.type !== "mesg") {
        return expr;
      }
      if (consumeToken(TOKEN_COLON)) {
        return {
          type: "narg",
          name: expr.name,
          value: parseLiteral()
        };
      }
      return expr;
    }
    function parseVariants() {
      let variants = [];
      let count = 0;
      let star;
      while (test(RE_VARIANT_START)) {
        if (consumeChar("*")) {
          star = count;
        }
        let key = parseVariantKey();
        let value = parsePattern();
        if (value === null) {
          throw new SyntaxError("Expected variant value");
        }
        variants[count++] = {
          key,
          value
        };
      }
      if (count === 0) {
        return null;
      }
      if (star === undefined) {
        throw new SyntaxError("Expected default variant");
      }
      return {
        variants,
        star
      };
    }
    function parseVariantKey() {
      consumeToken(TOKEN_BRACKET_OPEN, SyntaxError);
      let key;
      if (test(RE_NUMBER_LITERAL)) {
        key = parseNumberLiteral();
      } else {
        key = {
          type: "str",
          value: match1(RE_IDENTIFIER)
        };
      }
      consumeToken(TOKEN_BRACKET_CLOSE, SyntaxError);
      return key;
    }
    function parseLiteral() {
      if (test(RE_NUMBER_LITERAL)) {
        return parseNumberLiteral();
      }
      if (source[cursor] === '"') {
        return parseStringLiteral();
      }
      throw new SyntaxError("Invalid expression");
    }
    function parseNumberLiteral() {
      let [, value, fraction = ""] = match(RE_NUMBER_LITERAL);
      let precision = fraction.length;
      return {
        type: "num",
        value: parseFloat(value),
        precision
      };
    }
    function parseStringLiteral() {
      consumeChar('"', SyntaxError);
      let value = "";
      while (true) {
        value += match1(RE_STRING_RUN);
        if (source[cursor] === "\\") {
          value += parseEscapeSequence();
          continue;
        }
        if (consumeChar('"')) {
          return {
            type: "str",
            value
          };
        }
        throw new SyntaxError("Unclosed string literal");
      }
    }
    function parseEscapeSequence() {
      if (test(RE_STRING_ESCAPE)) {
        return match1(RE_STRING_ESCAPE);
      }
      if (test(RE_UNICODE_ESCAPE)) {
        let [, codepoint4, codepoint6] = match(RE_UNICODE_ESCAPE);
        let codepoint = parseInt(codepoint4 || codepoint6, 16);
        return codepoint <= 0xd7ff || 0xe000 <= codepoint ? String.fromCodePoint(codepoint) : "�";
      }
      throw new SyntaxError("Unknown escape sequence");
    }
    function parseIndent() {
      let start = cursor;
      consumeToken(TOKEN_BLANK);
      switch (source[cursor]) {
        case ".":
        case "[":
        case "*":
        case "}":
        case undefined:
          return false;
        case "{":
          return makeIndent(source.slice(start, cursor));
      }
      if (source[cursor - 1] === " ") {
        return makeIndent(source.slice(start, cursor));
      }
      return false;
    }
    function trim(text, re) {
      return text.replace(re, "");
    }
    function makeIndent(blank) {
      let value = blank.replace(RE_BLANK_LINES, "\n");
      let length = RE_INDENT.exec(blank)[1].length;
      return new Indent(value, length);
    }
  }
}
class Indent {
  constructor(value, length) {
    this.value = value;
    this.length = length;
  }
}
;// ./node_modules/@fluent/bundle/esm/index.js



;// ./node_modules/@fluent/dom/esm/overlay.js



const reOverlay = /<|&#?\w+;/;
const TEXT_LEVEL_ELEMENTS = {
  "http://www.w3.org/1999/xhtml": ["em", "strong", "small", "s", "cite", "q", "dfn", "abbr", "data", "time", "code", "var", "samp", "kbd", "sub", "sup", "i", "b", "u", "mark", "bdi", "bdo", "span", "br", "wbr"]
};
const LOCALIZABLE_ATTRIBUTES = {
  "http://www.w3.org/1999/xhtml": {
    global: ["title", "aria-description", "aria-label", "aria-valuetext"],
    a: ["download"],
    area: ["download", "alt"],
    input: ["alt", "placeholder"],
    menuitem: ["label"],
    menu: ["label"],
    optgroup: ["label"],
    option: ["label"],
    track: ["label"],
    img: ["alt"],
    textarea: ["placeholder"],
    th: ["abbr"]
  },
  "http://www.mozilla.org/keymaster/gatekeeper/there.is.only.xul": {
    global: ["accesskey", "aria-label", "aria-valuetext", "label", "title", "tooltiptext"],
    description: ["value"],
    key: ["key", "keycode"],
    label: ["value"],
    textbox: ["placeholder", "value"]
  }
};
function translateElement(element, translation) {
  const {
    value
  } = translation;
  if (typeof value === "string") {
    if (element.localName === "title" && element.namespaceURI === "http://www.w3.org/1999/xhtml") {
      element.textContent = value;
    } else if (!reOverlay.test(value)) {
      element.textContent = value;
    } else {
      const templateElement = element.ownerDocument.createElementNS("http://www.w3.org/1999/xhtml", "template");
      templateElement.innerHTML = value;
      overlayChildNodes(templateElement.content, element);
    }
  }
  overlayAttributes(translation, element);
}
function overlayChildNodes(fromFragment, toElement) {
  for (const childNode of fromFragment.childNodes) {
    if (childNode.nodeType === childNode.TEXT_NODE) {
      continue;
    }
    if (childNode.hasAttribute("data-l10n-name")) {
      const sanitized = getNodeForNamedElement(toElement, childNode);
      fromFragment.replaceChild(sanitized, childNode);
      continue;
    }
    if (isElementAllowed(childNode)) {
      const sanitized = createSanitizedElement(childNode);
      fromFragment.replaceChild(sanitized, childNode);
      continue;
    }
    console.warn(`An element of forbidden type "${childNode.localName}" was found in ` + "the translation. Only safe text-level elements and elements with " + "data-l10n-name are allowed.");
    fromFragment.replaceChild(createTextNodeFromTextContent(childNode), childNode);
  }
  toElement.textContent = "";
  toElement.appendChild(fromFragment);
}
function hasAttribute(attributes, name) {
  if (!attributes) {
    return false;
  }
  for (let attr of attributes) {
    if (attr.name === name) {
      return true;
    }
  }
  return false;
}
function overlayAttributes(fromElement, toElement) {
  const explicitlyAllowed = toElement.hasAttribute("data-l10n-attrs") ? toElement.getAttribute("data-l10n-attrs").split(",").map(i => i.trim()) : null;
  for (const attr of Array.from(toElement.attributes)) {
    if (isAttrNameLocalizable(attr.name, toElement, explicitlyAllowed) && !hasAttribute(fromElement.attributes, attr.name)) {
      toElement.removeAttribute(attr.name);
    }
  }
  if (!fromElement.attributes) {
    return;
  }
  for (const attr of Array.from(fromElement.attributes)) {
    if (isAttrNameLocalizable(attr.name, toElement, explicitlyAllowed) && toElement.getAttribute(attr.name) !== attr.value) {
      toElement.setAttribute(attr.name, attr.value);
    }
  }
}
function getNodeForNamedElement(sourceElement, translatedChild) {
  const childName = translatedChild.getAttribute("data-l10n-name");
  const sourceChild = sourceElement.querySelector(`[data-l10n-name="${childName}"]`);
  if (!sourceChild) {
    console.warn(`An element named "${childName}" wasn't found in the source.`);
    return createTextNodeFromTextContent(translatedChild);
  }
  if (sourceChild.localName !== translatedChild.localName) {
    console.warn(`An element named "${childName}" was found in the translation ` + `but its type ${translatedChild.localName} didn't match the ` + `element found in the source (${sourceChild.localName}).`);
    return createTextNodeFromTextContent(translatedChild);
  }
  sourceElement.removeChild(sourceChild);
  const clone = sourceChild.cloneNode(false);
  return shallowPopulateUsing(translatedChild, clone);
}
function createSanitizedElement(element) {
  const clone = element.ownerDocument.createElement(element.localName);
  return shallowPopulateUsing(element, clone);
}
function createTextNodeFromTextContent(element) {
  return element.ownerDocument.createTextNode(element.textContent);
}
function isElementAllowed(element) {
  const allowed = TEXT_LEVEL_ELEMENTS[element.namespaceURI];
  return allowed && allowed.includes(element.localName);
}
function isAttrNameLocalizable(name, element, explicitlyAllowed = null) {
  if (explicitlyAllowed && explicitlyAllowed.includes(name)) {
    return true;
  }
  const allowed = LOCALIZABLE_ATTRIBUTES[element.namespaceURI];
  if (!allowed) {
    return false;
  }
  const attrName = name.toLowerCase();
  const elemName = element.localName;
  if (allowed.global.includes(attrName)) {
    return true;
  }
  if (!allowed[elemName]) {
    return false;
  }
  if (allowed[elemName].includes(attrName)) {
    return true;
  }
  if (element.namespaceURI === "http://www.w3.org/1999/xhtml" && elemName === "input" && attrName === "value") {
    const type = element.type.toLowerCase();
    if (type === "submit" || type === "button" || type === "reset") {
      return true;
    }
  }
  return false;
}
function shallowPopulateUsing(fromElement, toElement) {
  toElement.textContent = fromElement.textContent;
  overlayAttributes(fromElement, toElement);
  return toElement;
}
;// ./node_modules/cached-iterable/src/cached_iterable.mjs
class CachedIterable extends Array {
  static from(iterable) {
    if (iterable instanceof this) {
      return iterable;
    }
    return new this(iterable);
  }
}
;// ./node_modules/cached-iterable/src/cached_sync_iterable.mjs


class CachedSyncIterable extends CachedIterable {
  constructor(iterable) {
    super();
    if (Symbol.iterator in Object(iterable)) {
      this.iterator = iterable[Symbol.iterator]();
    } else {
      throw new TypeError("Argument must implement the iteration protocol.");
    }
  }
  [Symbol.iterator]() {
    const cached = this;
    let cur = 0;
    return {
      next() {
        if (cached.length <= cur) {
          cached.push(cached.iterator.next());
        }
        return cached[cur++];
      }
    };
  }
  touchNext(count = 1) {
    let idx = 0;
    while (idx++ < count) {
      const last = this[this.length - 1];
      if (last && last.done) {
        break;
      }
      this.push(this.iterator.next());
    }
    return this[this.length - 1];
  }
}
;// ./node_modules/cached-iterable/src/cached_async_iterable.mjs


class CachedAsyncIterable extends CachedIterable {
  constructor(iterable) {
    super();
    if (Symbol.asyncIterator in Object(iterable)) {
      this.iterator = iterable[Symbol.asyncIterator]();
    } else if (Symbol.iterator in Object(iterable)) {
      this.iterator = iterable[Symbol.iterator]();
    } else {
      throw new TypeError("Argument must implement the iteration protocol.");
    }
  }
  [Symbol.asyncIterator]() {
    const cached = this;
    let cur = 0;
    return {
      async next() {
        if (cached.length <= cur) {
          cached.push(cached.iterator.next());
        }
        return cached[cur++];
      }
    };
  }
  async touchNext(count = 1) {
    let idx = 0;
    while (idx++ < count) {
      const last = this[this.length - 1];
      if (last && (await last).done) {
        break;
      }
      this.push(this.iterator.next());
    }
    return this[this.length - 1];
  }
}
;// ./node_modules/cached-iterable/src/index.mjs


;// ./node_modules/@fluent/dom/esm/localization.js














class Localization {
  constructor(resourceIds = [], generateBundles) {
    this.resourceIds = resourceIds;
    this.generateBundles = generateBundles;
    this.onChange(true);
  }
  addResourceIds(resourceIds, eager = false) {
    this.resourceIds.push(...resourceIds);
    this.onChange(eager);
    return this.resourceIds.length;
  }
  removeResourceIds(resourceIds) {
    this.resourceIds = this.resourceIds.filter(r => !resourceIds.includes(r));
    this.onChange();
    return this.resourceIds.length;
  }
  async formatWithFallback(keys, method) {
    const translations = [];
    let hasAtLeastOneBundle = false;
    for await (const bundle of this.bundles) {
      hasAtLeastOneBundle = true;
      const missingIds = keysFromBundle(method, bundle, keys, translations);
      if (missingIds.size === 0) {
        break;
      }
      if (typeof console !== "undefined") {
        const locale = bundle.locales[0];
        const ids = Array.from(missingIds).join(", ");
        console.warn(`[fluent] Missing translations in ${locale}: ${ids}`);
      }
    }
    if (!hasAtLeastOneBundle && typeof console !== "undefined") {
      console.warn(`[fluent] Request for keys failed because no resource bundles got generated.
  keys: ${JSON.stringify(keys)}.
  resourceIds: ${JSON.stringify(this.resourceIds)}.`);
    }
    return translations;
  }
  formatMessages(keys) {
    return this.formatWithFallback(keys, messageFromBundle);
  }
  formatValues(keys) {
    return this.formatWithFallback(keys, valueFromBundle);
  }
  async formatValue(id, args) {
    const [val] = await this.formatValues([{
      id,
      args
    }]);
    return val;
  }
  handleEvent() {
    this.onChange();
  }
  onChange(eager = false) {
    this.bundles = CachedAsyncIterable.from(this.generateBundles(this.resourceIds));
    if (eager) {
      this.bundles.touchNext(2);
    }
  }
}
function valueFromBundle(bundle, errors, message, args) {
  if (message.value) {
    return bundle.formatPattern(message.value, args, errors);
  }
  return null;
}
function messageFromBundle(bundle, errors, message, args) {
  const formatted = {
    value: null,
    attributes: null
  };
  if (message.value) {
    formatted.value = bundle.formatPattern(message.value, args, errors);
  }
  let attrNames = Object.keys(message.attributes);
  if (attrNames.length > 0) {
    formatted.attributes = new Array(attrNames.length);
    for (let [i, name] of attrNames.entries()) {
      let value = bundle.formatPattern(message.attributes[name], args, errors);
      formatted.attributes[i] = {
        name,
        value
      };
    }
  }
  return formatted;
}
function keysFromBundle(method, bundle, keys, translations) {
  const messageErrors = [];
  const missingIds = new Set();
  keys.forEach(({
    id,
    args
  }, i) => {
    if (translations[i] !== undefined) {
      return;
    }
    let message = bundle.getMessage(id);
    if (message) {
      messageErrors.length = 0;
      translations[i] = method(bundle, messageErrors, message, args);
      if (messageErrors.length > 0 && typeof console !== "undefined") {
        const locale = bundle.locales[0];
        const errors = messageErrors.join(", ");
        console.warn(`[fluent][resolver] errors in ${locale}/${id}: ${errors}.`);
      }
    } else {
      missingIds.add(id);
    }
  });
  return missingIds;
}
;// ./node_modules/@fluent/dom/esm/dom_localization.js














const L10NID_ATTR_NAME = "data-l10n-id";
const L10NARGS_ATTR_NAME = "data-l10n-args";
const L10N_ELEMENT_QUERY = `[${L10NID_ATTR_NAME}]`;
class DOMLocalization extends Localization {
  constructor(resourceIds, generateBundles) {
    super(resourceIds, generateBundles);
    this.roots = new Set();
    this.pendingrAF = null;
    this.pendingElements = new Set();
    this.windowElement = null;
    this.mutationObserver = null;
    this.observerConfig = {
      attributes: true,
      characterData: false,
      childList: true,
      subtree: true,
      attributeFilter: [L10NID_ATTR_NAME, L10NARGS_ATTR_NAME]
    };
  }
  onChange(eager = false) {
    super.onChange(eager);
    if (this.roots) {
      this.translateRoots();
    }
  }
  setAttributes(element, id, args) {
    element.setAttribute(L10NID_ATTR_NAME, id);
    if (args) {
      element.setAttribute(L10NARGS_ATTR_NAME, JSON.stringify(args));
    } else {
      element.removeAttribute(L10NARGS_ATTR_NAME);
    }
    return element;
  }
  getAttributes(element) {
    return {
      id: element.getAttribute(L10NID_ATTR_NAME),
      args: JSON.parse(element.getAttribute(L10NARGS_ATTR_NAME) || null)
    };
  }
  connectRoot(newRoot) {
    for (const root of this.roots) {
      if (root === newRoot || root.contains(newRoot) || newRoot.contains(root)) {
        throw new Error("Cannot add a root that overlaps with existing root.");
      }
    }
    if (this.windowElement) {
      if (this.windowElement !== newRoot.ownerDocument.defaultView) {
        throw new Error(`Cannot connect a root:
          DOMLocalization already has a root from a different window.`);
      }
    } else {
      this.windowElement = newRoot.ownerDocument.defaultView;
      this.mutationObserver = new this.windowElement.MutationObserver(mutations => this.translateMutations(mutations));
    }
    this.roots.add(newRoot);
    this.mutationObserver.observe(newRoot, this.observerConfig);
  }
  disconnectRoot(root) {
    this.roots.delete(root);
    this.pauseObserving();
    if (this.roots.size === 0) {
      this.mutationObserver = null;
      if (this.windowElement && this.pendingrAF) {
        this.windowElement.cancelAnimationFrame(this.pendingrAF);
      }
      this.windowElement = null;
      this.pendingrAF = null;
      this.pendingElements.clear();
      return true;
    }
    this.resumeObserving();
    return false;
  }
  translateRoots() {
    const roots = Array.from(this.roots);
    return Promise.all(roots.map(root => this.translateFragment(root)));
  }
  pauseObserving() {
    if (!this.mutationObserver) {
      return;
    }
    this.translateMutations(this.mutationObserver.takeRecords());
    this.mutationObserver.disconnect();
  }
  resumeObserving() {
    if (!this.mutationObserver) {
      return;
    }
    for (const root of this.roots) {
      this.mutationObserver.observe(root, this.observerConfig);
    }
  }
  translateMutations(mutations) {
    for (const mutation of mutations) {
      switch (mutation.type) {
        case "attributes":
          if (mutation.target.hasAttribute("data-l10n-id")) {
            this.pendingElements.add(mutation.target);
          }
          break;
        case "childList":
          for (const addedNode of mutation.addedNodes) {
            if (addedNode.nodeType === addedNode.ELEMENT_NODE) {
              if (addedNode.childElementCount) {
                for (const element of this.getTranslatables(addedNode)) {
                  this.pendingElements.add(element);
                }
              } else if (addedNode.hasAttribute(L10NID_ATTR_NAME)) {
                this.pendingElements.add(addedNode);
              }
            }
          }
          break;
      }
    }
    if (this.pendingElements.size > 0) {
      if (this.pendingrAF === null) {
        this.pendingrAF = this.windowElement.requestAnimationFrame(() => {
          this.translateElements(Array.from(this.pendingElements));
          this.pendingElements.clear();
          this.pendingrAF = null;
        });
      }
    }
  }
  translateFragment(frag) {
    return this.translateElements(this.getTranslatables(frag));
  }
  async translateElements(elements) {
    if (!elements.length) {
      return undefined;
    }
    const keys = elements.map(this.getKeysForElement);
    const translations = await this.formatMessages(keys);
    return this.applyTranslations(elements, translations);
  }
  applyTranslations(elements, translations) {
    this.pauseObserving();
    for (let i = 0; i < elements.length; i++) {
      if (translations[i] !== undefined) {
        translateElement(elements[i], translations[i]);
      }
    }
    this.resumeObserving();
  }
  getTranslatables(element) {
    const nodes = Array.from(element.querySelectorAll(L10N_ELEMENT_QUERY));
    if (typeof element.hasAttribute === "function" && element.hasAttribute(L10NID_ATTR_NAME)) {
      nodes.push(element);
    }
    return nodes;
  }
  getKeysForElement(element) {
    return {
      id: element.getAttribute(L10NID_ATTR_NAME),
      args: JSON.parse(element.getAttribute(L10NARGS_ATTR_NAME) || null)
    };
  }
}
;// ./node_modules/@fluent/dom/esm/index.js


;// ./web/l10n.js










class L10n {
  #dir;
  #elements;
  #lang;
  #l10n;
  constructor({
    lang,
    isRTL
  }, l10n = null) {
    this.#lang = L10n.#fixupLangCode(lang);
    this.#l10n = l10n;
    this.#dir = isRTL ?? L10n.#isRTL(this.#lang) ? "rtl" : "ltr";
  }
  _setL10n(l10n) {
    this.#l10n = l10n;
  }
  getLanguage() {
    return this.#lang;
  }
  getDirection() {
    return this.#dir;
  }
  async get(ids, args = null, fallback) {
    if (Array.isArray(ids)) {
      ids = ids.map(id => ({
        id
      }));
      const messages = await this.#l10n.formatMessages(ids);
      return messages.map(message => message.value);
    }
    const messages = await this.#l10n.formatMessages([{
      id: ids,
      args
    }]);
    return messages[0]?.value || fallback;
  }
  async translate(element) {
    (this.#elements ||= new Set()).add(element);
    try {
      this.#l10n.connectRoot(element);
      await this.#l10n.translateRoots();
    } catch {}
  }
  async translateOnce(element) {
    try {
      await this.#l10n.translateElements([element]);
    } catch (ex) {
      console.error("translateOnce:", ex);
    }
  }
  async destroy() {
    if (this.#elements) {
      for (const element of this.#elements) {
        this.#l10n.disconnectRoot(element);
      }
      this.#elements.clear();
      this.#elements = null;
    }
    this.#l10n.pauseObserving();
  }
  pause() {
    this.#l10n.pauseObserving();
  }
  resume() {
    this.#l10n.resumeObserving();
  }
  static #fixupLangCode(langCode) {
    langCode = langCode?.toLowerCase() || "en-us";
    const PARTIAL_LANG_CODES = {
      en: "en-us",
      es: "es-es",
      fy: "fy-nl",
      ga: "ga-ie",
      gu: "gu-in",
      hi: "hi-in",
      hy: "hy-am",
      nb: "nb-no",
      ne: "ne-np",
      nn: "nn-no",
      pa: "pa-in",
      pt: "pt-pt",
      sv: "sv-se",
      zh: "zh-cn"
    };
    return PARTIAL_LANG_CODES[langCode] || langCode;
  }
  static #isRTL(lang) {
    const shortCode = lang.split("-", 1)[0];
    return ["ar", "he", "fa", "ps", "ur"].includes(shortCode);
  }
}
const GenericL10n = null;

;// ./web/genericl10n.js





function PLATFORM() {
  const {
    isAndroid,
    isLinux,
    isMac,
    isWindows
  } = FeatureTest.platform;
  if (isLinux) {
    return "linux";
  }
  if (isWindows) {
    return "windows";
  }
  if (isMac) {
    return "macos";
  }
  if (isAndroid) {
    return "android";
  }
  return "other";
}
function createBundle(lang, text) {
  const resource = new FluentResource(text);
  const bundle = new FluentBundle(lang, {
    functions: {
      PLATFORM
    }
  });
  const errors = bundle.addResource(resource);
  if (errors.length) {
    console.error("L10n errors", errors);
  }
  return bundle;
}
class genericl10n_GenericL10n extends L10n {
  constructor(lang) {
    super({
      lang
    });
    const generateBundles = !lang ? genericl10n_GenericL10n.#generateBundlesFallback.bind(genericl10n_GenericL10n, this.getLanguage()) : genericl10n_GenericL10n.#generateBundles.bind(genericl10n_GenericL10n, "en-us", this.getLanguage());
    this._setL10n(new DOMLocalization([], generateBundles));
  }
  static async *#generateBundles(defaultLang, baseLang) {
    const {
      baseURL,
      paths
    } = await this.#getPaths();
    const langs = [baseLang];
    if (defaultLang !== baseLang) {
      const shortLang = baseLang.split("-", 1)[0];
      if (shortLang !== baseLang) {
        langs.push(shortLang);
      }
      langs.push(defaultLang);
    }
    const bundles = langs.map(lang => [lang, this.#createBundle(lang, baseURL, paths)]);
    for (const [lang, bundlePromise] of bundles) {
      const bundle = await bundlePromise;
      if (bundle) {
        yield bundle;
      } else if (lang === "en-us") {
        yield this.#createBundleFallback(lang);
      }
    }
  }
  static async #createBundle(lang, baseURL, paths) {
    const path = paths[lang];
    if (!path) {
      return null;
    }
    const url = new URL(path, baseURL);
    const text = await fetchData(url, "text");
    return createBundle(lang, text);
  }
  static async #getPaths() {
    try {
      const {
        href
      } = document.querySelector(`link[type="application/l10n"]`);
      const paths = await fetchData(href, "json");
      return {
        baseURL: href.substring(0, href.lastIndexOf("/") + 1) || "./",
        paths
      };
    } catch {}
    return {
      baseURL: "./",
      paths: Object.create(null)
    };
  }
  static async *#generateBundlesFallback(lang) {
    yield this.#createBundleFallback(lang);
  }
  static async #createBundleFallback(lang) {
    const text = "pdfjs-previous-button =\n    .title = Previous Page\npdfjs-previous-button-label = Previous\npdfjs-next-button =\n    .title = Next Page\npdfjs-next-button-label = Next\npdfjs-page-input =\n    .title = Page\npdfjs-of-pages = of { $pagesCount }\npdfjs-page-of-pages = ({ $pageNumber } of { $pagesCount })\npdfjs-zoom-out-button =\n    .title = Zoom Out\npdfjs-zoom-out-button-label = Zoom Out\npdfjs-zoom-in-button =\n    .title = Zoom In\npdfjs-zoom-in-button-label = Zoom In\npdfjs-zoom-select =\n    .title = Zoom\npdfjs-presentation-mode-button =\n    .title = Switch to Presentation Mode\npdfjs-presentation-mode-button-label = Presentation Mode\npdfjs-open-file-button =\n    .title = Open File\npdfjs-open-file-button-label = Open\npdfjs-print-button =\n    .title = Print\npdfjs-print-button-label = Print\npdfjs-save-button =\n    .title = Save\npdfjs-save-button-label = Save\npdfjs-download-button =\n    .title = Download\npdfjs-download-button-label = Download\npdfjs-bookmark-button =\n    .title = Current Page (View URL from Current Page)\npdfjs-bookmark-button-label = Current Page\npdfjs-tools-button =\n    .title = Tools\npdfjs-tools-button-label = Tools\npdfjs-first-page-button =\n    .title = Go to First Page\npdfjs-first-page-button-label = Go to First Page\npdfjs-last-page-button =\n    .title = Go to Last Page\npdfjs-last-page-button-label = Go to Last Page\npdfjs-page-rotate-cw-button =\n    .title = Rotate Clockwise\npdfjs-page-rotate-cw-button-label = Rotate Clockwise\npdfjs-page-rotate-ccw-button =\n    .title = Rotate Counterclockwise\npdfjs-page-rotate-ccw-button-label = Rotate Counterclockwise\npdfjs-cursor-text-select-tool-button =\n    .title = Enable Text Selection Tool\npdfjs-cursor-text-select-tool-button-label = Text Selection Tool\npdfjs-cursor-hand-tool-button =\n    .title = Enable Hand Tool\npdfjs-cursor-hand-tool-button-label = Hand Tool\npdfjs-scroll-page-button =\n    .title = Use Page Scrolling\npdfjs-scroll-page-button-label = Page Scrolling\npdfjs-scroll-vertical-button =\n    .title = Use Vertical Scrolling\npdfjs-scroll-vertical-button-label = Vertical Scrolling\npdfjs-scroll-horizontal-button =\n    .title = Use Horizontal Scrolling\npdfjs-scroll-horizontal-button-label = Horizontal Scrolling\npdfjs-scroll-wrapped-button =\n    .title = Use Wrapped Scrolling\npdfjs-scroll-wrapped-button-label = Wrapped Scrolling\npdfjs-spread-none-button =\n    .title = Do not join page spreads\npdfjs-spread-none-button-label = No Spreads\npdfjs-spread-odd-button =\n    .title = Join page spreads starting with odd-numbered pages\npdfjs-spread-odd-button-label = Odd Spreads\npdfjs-spread-even-button =\n    .title = Join page spreads starting with even-numbered pages\npdfjs-spread-even-button-label = Even Spreads\npdfjs-document-properties-button =\n    .title = Document Properties…\npdfjs-document-properties-button-label = Document Properties…\npdfjs-document-properties-file-name = File name:\npdfjs-document-properties-file-size = File size:\npdfjs-document-properties-size-kb = { NUMBER($kb, maximumSignificantDigits: 3) } KB ({ $b } bytes)\npdfjs-document-properties-size-mb = { NUMBER($mb, maximumSignificantDigits: 3) } MB ({ $b } bytes)\npdfjs-document-properties-title = Title:\npdfjs-document-properties-author = Author:\npdfjs-document-properties-subject = Subject:\npdfjs-document-properties-keywords = Keywords:\npdfjs-document-properties-creation-date = Creation Date:\npdfjs-document-properties-modification-date = Modification Date:\npdfjs-document-properties-date-time-string = { DATETIME($dateObj, dateStyle: \"short\", timeStyle: \"medium\") }\npdfjs-document-properties-creator = Creator:\npdfjs-document-properties-producer = PDF Producer:\npdfjs-document-properties-version = PDF Version:\npdfjs-document-properties-page-count = Page Count:\npdfjs-document-properties-page-size = Page Size:\npdfjs-document-properties-page-size-unit-inches = in\npdfjs-document-properties-page-size-unit-millimeters = mm\npdfjs-document-properties-page-size-orientation-portrait = portrait\npdfjs-document-properties-page-size-orientation-landscape = landscape\npdfjs-document-properties-page-size-name-a-three = A3\npdfjs-document-properties-page-size-name-a-four = A4\npdfjs-document-properties-page-size-name-letter = Letter\npdfjs-document-properties-page-size-name-legal = Legal\npdfjs-document-properties-page-size-dimension-string = { $width } × { $height } { $unit } ({ $orientation })\npdfjs-document-properties-page-size-dimension-name-string = { $width } × { $height } { $unit } ({ $name }, { $orientation })\npdfjs-document-properties-linearized = Fast Web View:\npdfjs-document-properties-linearized-yes = Yes\npdfjs-document-properties-linearized-no = No\npdfjs-document-properties-close-button = Close\npdfjs-print-progress-message = Preparing document for printing…\npdfjs-print-progress-percent = { $progress }%\npdfjs-print-progress-close-button = Cancel\npdfjs-printing-not-supported = Warning: Printing is not fully supported by this browser.\npdfjs-printing-not-ready = Warning: The PDF is not fully loaded for printing.\npdfjs-current-outline-item-button =\n    .title = Find Current Outline Item\npdfjs-current-outline-item-button-label = Current Outline Item\npdfjs-findbar-button =\n    .title = Find in Document\npdfjs-findbar-button-label = Find\npdfjs-additional-layers = Additional Layers\npdfjs-thumb-page-title1 =\n    .title = Page { $page } of { $total }\npdfjs-thumb-page-canvas =\n    .aria-label = Thumbnail of Page { $page }\npdfjs-thumb-page-checkbox1 =\n    .title = Select page { $page }\npdfjs-find-input =\n    .title = Find\n    .placeholder = Find in document…\npdfjs-find-previous-button =\n    .title = Find the previous occurrence of the phrase\npdfjs-find-previous-button-label = Previous\npdfjs-find-next-button =\n    .title = Find the next occurrence of the phrase\npdfjs-find-next-button-label = Next\npdfjs-find-highlight-checkbox = Highlight All\npdfjs-find-match-case-checkbox-label = Match Case\npdfjs-find-match-diacritics-checkbox-label = Match Diacritics\npdfjs-find-entire-word-checkbox-label = Whole Words\npdfjs-find-reached-top = Reached top of document, continued from bottom\npdfjs-find-reached-bottom = Reached end of document, continued from top\npdfjs-find-match-count =\n    { $total ->\n        [one] { $current } of { $total } match\n       *[other] { $current } of { $total } matches\n    }\npdfjs-find-match-count-limit =\n    { $limit ->\n        [one] More than { $limit } match\n       *[other] More than { $limit } matches\n    }\npdfjs-find-not-found = Phrase not found\npdfjs-page-scale-width = Page Width\npdfjs-page-scale-fit = Page Fit\npdfjs-page-scale-auto = Automatic Zoom\npdfjs-page-scale-actual = Actual Size\npdfjs-page-scale-percent = { $scale }%\npdfjs-page-landmark =\n    .aria-label = Page { $page }\npdfjs-loading-error = An error occurred while loading the PDF.\npdfjs-invalid-file-error = Invalid or corrupted PDF file.\npdfjs-missing-file-error = Missing PDF file.\npdfjs-unexpected-response-error = Unexpected server response.\npdfjs-rendering-error = An error occurred while rendering the page.\npdfjs-annotation-date-time-string = { DATETIME($dateObj, dateStyle: \"short\", timeStyle: \"medium\") }\npdfjs-text-annotation-type =\n    .alt = [{ $type } Annotation]\npdfjs-password-label = Enter the password to open this PDF file.\npdfjs-password-invalid = Invalid password. Please try again.\npdfjs-password-ok-button = OK\npdfjs-password-cancel-button = Cancel\npdfjs-web-fonts-disabled = Web fonts are disabled: unable to use embedded PDF fonts.\npdfjs-editor-free-text-button =\n    .title = Text\npdfjs-editor-color-picker-free-text-input =\n    .title = Change text color\npdfjs-editor-free-text-button-label = Text\npdfjs-editor-ink-button =\n    .title = Draw\npdfjs-editor-color-picker-ink-input =\n    .title = Change drawing color\npdfjs-editor-ink-button-label = Draw\npdfjs-editor-stamp-button =\n    .title = Add or edit images\npdfjs-editor-stamp-button-label = Add or edit images\npdfjs-editor-highlight-button =\n    .title = Highlight\npdfjs-editor-highlight-button-label = Highlight\npdfjs-highlight-floating-button1 =\n    .title = Highlight\n    .aria-label = Highlight\npdfjs-highlight-floating-button-label = Highlight\npdfjs-comment-floating-button =\n    .title = Comment\n    .aria-label = Comment\npdfjs-comment-floating-button-label = Comment\npdfjs-editor-comment-button =\n    .title = Comment\n    .aria-label = Comment\npdfjs-editor-comment-button-label = Comment\npdfjs-editor-signature-button =\n    .title = Add signature\npdfjs-editor-signature-button-label = Add signature\npdfjs-editor-highlight-editor =\n    .aria-label = Highlight editor\npdfjs-editor-ink-editor =\n    .aria-label = Drawing editor\npdfjs-editor-signature-editor1 =\n    .aria-description = Signature editor: { $description }\npdfjs-editor-stamp-editor =\n    .aria-label = Image editor\npdfjs-editor-remove-ink-button =\n    .title = Remove drawing\npdfjs-editor-remove-freetext-button =\n    .title = Remove text\npdfjs-editor-remove-stamp-button =\n    .title = Remove image\npdfjs-editor-remove-highlight-button =\n    .title = Remove highlight\npdfjs-editor-remove-signature-button =\n    .title = Remove signature\npdfjs-editor-free-text-color-input = Color\npdfjs-editor-free-text-size-input = Size\npdfjs-editor-ink-color-input = Color\npdfjs-editor-ink-thickness-input = Thickness\npdfjs-editor-ink-opacity-input = Opacity\npdfjs-editor-stamp-add-image-button =\n    .title = Add image\npdfjs-editor-stamp-add-image-button-label = Add image\npdfjs-editor-free-highlight-thickness-input = Thickness\npdfjs-editor-free-highlight-thickness-title =\n    .title = Change thickness when highlighting items other than text\npdfjs-editor-add-signature-container =\n    .aria-label = Signature controls and saved signatures\npdfjs-editor-signature-add-signature-button =\n    .title = Add new signature\npdfjs-editor-signature-add-signature-button-label = Add new signature\npdfjs-editor-add-saved-signature-button =\n    .title = Saved signature: { $description }\npdfjs-free-text2 =\n    .aria-label = Text Editor\n    .default-content = Start typing…\npdfjs-editor-comments-sidebar-title =\n    { $count ->\n        [one] Comment\n       *[other] Comments\n    }\npdfjs-editor-comments-sidebar-close-button =\n    .title = Close the sidebar\n    .aria-label = Close the sidebar\npdfjs-editor-comments-sidebar-close-button-label = Close the sidebar\npdfjs-editor-comments-sidebar-no-comments1 = See something noteworthy? Highlight it and leave a comment.\npdfjs-editor-comments-sidebar-no-comments-link = Learn more\npdfjs-editor-alt-text-button =\n    .aria-label = Alt text\npdfjs-editor-alt-text-button-label = Alt text\npdfjs-editor-alt-text-edit-button =\n    .aria-label = Edit alt text\npdfjs-editor-alt-text-dialog-label = Choose an option\npdfjs-editor-alt-text-dialog-description = Alt text (alternative text) helps when people can’t see the image or when it doesn’t load.\npdfjs-editor-alt-text-add-description-label = Add a description\npdfjs-editor-alt-text-add-description-description = Aim for 1-2 sentences that describe the subject, setting, or actions.\npdfjs-editor-alt-text-mark-decorative-label = Mark as decorative\npdfjs-editor-alt-text-mark-decorative-description = This is used for ornamental images, like borders or watermarks.\npdfjs-editor-alt-text-cancel-button = Cancel\npdfjs-editor-alt-text-save-button = Save\npdfjs-editor-alt-text-decorative-tooltip = Marked as decorative\npdfjs-editor-alt-text-textarea =\n    .placeholder = For example, “A young man sits down at a table to eat a meal”\npdfjs-editor-resizer-top-left =\n    .aria-label = Top left corner — resize\npdfjs-editor-resizer-top-middle =\n    .aria-label = Top middle — resize\npdfjs-editor-resizer-top-right =\n    .aria-label = Top right corner — resize\npdfjs-editor-resizer-middle-right =\n    .aria-label = Middle right — resize\npdfjs-editor-resizer-bottom-right =\n    .aria-label = Bottom right corner — resize\npdfjs-editor-resizer-bottom-middle =\n    .aria-label = Bottom middle — resize\npdfjs-editor-resizer-bottom-left =\n    .aria-label = Bottom left corner — resize\npdfjs-editor-resizer-middle-left =\n    .aria-label = Middle left — resize\npdfjs-editor-highlight-colorpicker-label = Highlight color\npdfjs-editor-colorpicker-button =\n    .title = Change color\npdfjs-editor-colorpicker-dropdown =\n    .aria-label = Color choices\npdfjs-editor-colorpicker-yellow =\n    .title = Yellow\npdfjs-editor-colorpicker-green =\n    .title = Green\npdfjs-editor-colorpicker-blue =\n    .title = Blue\npdfjs-editor-colorpicker-pink =\n    .title = Pink\npdfjs-editor-colorpicker-red =\n    .title = Red\npdfjs-editor-highlight-show-all-button-label = Show all\npdfjs-editor-highlight-show-all-button =\n    .title = Show all\npdfjs-editor-new-alt-text-dialog-edit-label = Edit alt text (image description)\npdfjs-editor-new-alt-text-dialog-add-label = Add alt text (image description)\npdfjs-editor-new-alt-text-textarea =\n    .placeholder = Write your description here…\npdfjs-editor-new-alt-text-description = Short description for people who can’t see the image or when the image doesn’t load.\npdfjs-editor-new-alt-text-disclaimer1 = This alt text was created automatically and may be inaccurate.\npdfjs-editor-new-alt-text-disclaimer-learn-more-url = Learn more\npdfjs-editor-new-alt-text-create-automatically-button-label = Create alt text automatically\npdfjs-editor-new-alt-text-not-now-button = Not now\npdfjs-editor-new-alt-text-error-title = Couldn’t create alt text automatically\npdfjs-editor-new-alt-text-error-description = Please write your own alt text or try again later.\npdfjs-editor-new-alt-text-error-close-button = Close\npdfjs-editor-new-alt-text-ai-model-downloading-progress = Downloading alt text AI model ({ $downloadedSize } of { $totalSize } MB)\n    .aria-valuetext = Downloading alt text AI model ({ $downloadedSize } of { $totalSize } MB)\npdfjs-editor-new-alt-text-added-button =\n    .aria-label = Alt text added\npdfjs-editor-new-alt-text-added-button-label = Alt text added\npdfjs-editor-new-alt-text-missing-button =\n    .aria-label = Missing alt text\npdfjs-editor-new-alt-text-missing-button-label = Missing alt text\npdfjs-editor-new-alt-text-to-review-button =\n    .aria-label = Review alt text\npdfjs-editor-new-alt-text-to-review-button-label = Review alt text\npdfjs-editor-new-alt-text-generated-alt-text-with-disclaimer = Created automatically: { $generatedAltText }\npdfjs-image-alt-text-settings-button =\n    .title = Image alt text settings\npdfjs-image-alt-text-settings-button-label = Image alt text settings\npdfjs-editor-alt-text-settings-dialog-label = Image alt text settings\npdfjs-editor-alt-text-settings-automatic-title = Automatic alt text\npdfjs-editor-alt-text-settings-create-model-button-label = Create alt text automatically\npdfjs-editor-alt-text-settings-create-model-description = Suggests descriptions to help people who can’t see the image or when the image doesn’t load.\npdfjs-editor-alt-text-settings-editor-title = Alt text editor\npdfjs-editor-alt-text-settings-show-dialog-button-label = Show alt text editor right away when adding an image\npdfjs-editor-alt-text-settings-show-dialog-description = Helps you make sure all your images have alt text.\npdfjs-editor-alt-text-settings-close-button = Close\npdfjs-editor-highlight-added-alert = Highlight added\npdfjs-editor-freetext-added-alert = Text added\npdfjs-editor-ink-added-alert = Drawing added\npdfjs-editor-stamp-added-alert = Image added\npdfjs-editor-signature-added-alert = Signature added\npdfjs-editor-undo-bar-message-highlight = Highlight removed\npdfjs-editor-undo-bar-message-freetext = Text removed\npdfjs-editor-undo-bar-message-ink = Drawing removed\npdfjs-editor-undo-bar-message-stamp = Image removed\npdfjs-editor-undo-bar-message-signature = Signature removed\npdfjs-editor-undo-bar-message-comment = Comment removed\npdfjs-editor-undo-bar-message-multiple =\n    { $count ->\n        [one] { $count } annotation removed\n       *[other] { $count } annotations removed\n    }\npdfjs-editor-undo-bar-undo-button =\n    .title = Undo\npdfjs-editor-undo-bar-undo-button-label = Undo\npdfjs-editor-undo-bar-close-button =\n    .title = Close\npdfjs-editor-undo-bar-close-button-label = Close\npdfjs-editor-add-signature-dialog-label = This modal allows the user to create a signature to add to a PDF document. The user can edit the name (which also serves as the alt text), and optionally save the signature for repeated use.\npdfjs-editor-add-signature-dialog-title = Add a signature\npdfjs-editor-add-signature-type-button = Type\n    .title = Type\npdfjs-editor-add-signature-draw-button = Draw\n    .title = Draw\npdfjs-editor-add-signature-image-button = Image\n    .title = Image\npdfjs-editor-add-signature-type-input =\n    .aria-label = Type your signature\n    .placeholder = Type your signature\npdfjs-editor-add-signature-draw-placeholder = Draw your signature\npdfjs-editor-add-signature-draw-thickness-range-label = Thickness\npdfjs-editor-add-signature-draw-thickness-range =\n    .title = Drawing thickness: { $thickness }\npdfjs-editor-add-signature-image-placeholder = Drag a file here to upload\npdfjs-editor-add-signature-image-browse-link =\n    { PLATFORM() ->\n        [macos] Or choose image files\n       *[other] Or browse image files\n    }\npdfjs-editor-add-signature-description-label = Description (alt text)\npdfjs-editor-add-signature-description-input =\n    .title = Description (alt text)\npdfjs-editor-add-signature-description-default-when-drawing = Signature\npdfjs-editor-add-signature-clear-button-label = Clear signature\npdfjs-editor-add-signature-clear-button =\n    .title = Clear signature\npdfjs-editor-add-signature-save-checkbox = Save signature\npdfjs-editor-add-signature-save-warning-message = You’ve reached the limit of 5 saved signatures. Remove one to save more.\npdfjs-editor-add-signature-image-upload-error-title = Couldn’t upload image\npdfjs-editor-add-signature-image-upload-error-description = Check your network connection or try another image.\npdfjs-editor-add-signature-image-no-data-error-title = Can’t convert this image into a signature\npdfjs-editor-add-signature-image-no-data-error-description = Please try uploading a different image.\npdfjs-editor-add-signature-error-close-button = Close\npdfjs-editor-add-signature-cancel-button = Cancel\npdfjs-editor-add-signature-add-button = Add\npdfjs-editor-delete-signature-button1 =\n    .title = Remove saved signature\npdfjs-editor-delete-signature-button-label1 = Remove saved signature\npdfjs-editor-add-signature-edit-button-label = Edit description\npdfjs-editor-edit-signature-dialog-title = Edit description\npdfjs-editor-edit-signature-update-button = Update\npdfjs-show-comment-button =\n    .title = Show comment\npdfjs-editor-edit-comment-popup-button-label = Edit comment\npdfjs-editor-edit-comment-popup-button =\n    .title = Edit comment\npdfjs-editor-delete-comment-popup-button-label = Remove comment\npdfjs-editor-delete-comment-popup-button =\n    .title = Remove comment\npdfjs-editor-edit-comment-dialog-title-when-editing = Edit comment\npdfjs-editor-edit-comment-dialog-save-button-when-editing = Update\npdfjs-editor-edit-comment-dialog-title-when-adding = Add comment\npdfjs-editor-edit-comment-dialog-save-button-when-adding = Add\npdfjs-editor-edit-comment-dialog-text-input =\n    .placeholder = Start typing…\npdfjs-editor-edit-comment-dialog-cancel-button = Cancel\npdfjs-editor-add-comment-button =\n    .title = Add comment\npdfjs-toggle-views-manager-button1 =\n    .title = Manage pages\npdfjs-toggle-views-manager-notification-button =\n    .title = Toggle Sidebar (document contains thumbnails/outline/attachments/layers)\npdfjs-toggle-views-manager-button1-label = Manage pages\npdfjs-views-manager-sidebar =\n    .aria-label = Sidebar\npdfjs-views-manager-sidebar-resizer =\n    .aria-label = Sidebar resizer\npdfjs-views-manager-view-selector-button =\n    .title = Views\npdfjs-views-manager-view-selector-button-label = Views\npdfjs-views-manager-pages-title = Pages\npdfjs-views-manager-outlines-title1 = Document outline\n    .title = Document outline (double-click to expand/collapse all items)\npdfjs-views-manager-attachments-title = Attachments\npdfjs-views-manager-layers-title1 = Layers\n    .title = Layers (double-click to reset all layers to the default state)\npdfjs-views-manager-pages-option-label = Pages\npdfjs-views-manager-outlines-option-label = Document outline\npdfjs-views-manager-attachments-option-label = Attachments\npdfjs-views-manager-layers-option-label = Layers\npdfjs-views-manager-add-file-button =\n    .title = Add file\npdfjs-views-manager-add-file-button-label = Add file\npdfjs-views-manager-pages-status-action-label =\n    { $count ->\n        [one] { $count } selected\n        *[other] { $count } selected\n    }\npdfjs-views-manager-pages-status-none-action-label = Select pages\npdfjs-views-manager-pages-status-action-button-label = Manage\npdfjs-views-manager-pages-status-copy-button-label = Copy\npdfjs-views-manager-pages-status-cut-button-label = Cut\npdfjs-views-manager-pages-status-delete-button-label = Delete\npdfjs-views-manager-pages-status-export-selected-button-label = Export selected…\npdfjs-views-manager-status-undo-cut-label =\n    { $count ->\n        [one] 1 page cut\n        *[other] { $count } pages cut\n    }\npdfjs-views-manager-pages-status-undo-copy-label =\n    { $count ->\n        [one] 1 page copied\n        *[other] { $count } pages copied\n    }\npdfjs-views-manager-pages-status-undo-delete-label =\n    { $count ->\n        [one] 1 page deleted\n        *[other] { $count } pages deleted\n    }\npdfjs-views-manager-status-undo-button-label = Undo\npdfjs-views-manager-status-done-button-label = Done\npdfjs-views-manager-status-close-button =\n    .title = Close\npdfjs-views-manager-status-close-button-label = Close\npdfjs-views-manager-paste-button-label = Paste\npdfjs-views-manager-paste-button-before =\n    .title = Paste before the first page\npdfjs-views-manager-paste-button-after =\n    .title = Paste after page { $page }\npdfjs-new-badge-content = NEW\npdfjs-views-manager-waiting-for-file = Uploading file…\npdfjs-digital-signature-properties-button =\n    .title = Digital signature properties\n    .aria-label = Digital signature properties\npdfjs-digital-signature-properties-button-label = Digital signature properties\npdfjs-digital-signature-properties-banner-verified = Document was signed with a valid digital signature\npdfjs-digital-signature-properties-banner-unknown =\n    { $count ->\n        [one] Document signed but { $count } digital signature could not be verified\n       *[other] Document signed but { $count } digital signatures could not be verified\n    }\npdfjs-digital-signature-properties-banner-untrusted =\n    { $count ->\n        [one] Document signed with { $count } certificate that is not trusted\n       *[other] Document signed with { $count } certificates that are not trusted\n    }\npdfjs-digital-signature-properties-banner-expired =\n    { $count ->\n        [one] Document signed with { $count } expired certificate\n       *[other] Document signed with { $count } expired certificates\n    }\npdfjs-digital-signature-properties-banner-invalid =\n    { $count ->\n        [one] Document has { $count } invalid digital signature\n       *[other] Document has { $count } invalid digital signatures\n    }\npdfjs-digital-signature-properties-banner-revoked =\n    { $count ->\n        [one] Document signed with { $count } revoked certificate\n       *[other] Document signed with { $count } revoked certificates\n    }\npdfjs-digital-signature-properties-status-verified = Status: Signature verified\npdfjs-digital-signature-properties-status-invalid = Status: Signature invalid\npdfjs-digital-signature-properties-status-unknown = Status: Unable to verify (unsupported)\npdfjs-digital-signature-properties-certificate-trusted = Certificate: Trusted ({ $issuer })\npdfjs-digital-signature-properties-certificate-unknown = Certificate: Unavailable\npdfjs-digital-signature-properties-certificate-untrusted = Certificate: Untrusted\npdfjs-digital-signature-properties-certificate-untrusted-unknown-issuer = Certificate: Unknown issuer ({ $issuer })\npdfjs-digital-signature-properties-certificate-untrusted-self-signed = Certificate: Self-signed ({ $issuer })\npdfjs-digital-signature-properties-certificate-untrusted-untrusted-issuer = Certificate: Untrusted issuer ({ $issuer })\npdfjs-digital-signature-properties-certificate-expired = Certificate: Expired\npdfjs-digital-signature-properties-certificate-expired-with-date = Certificate: Expired ({ DATETIME($dateObj, dateStyle: \"medium\") })\npdfjs-digital-signature-properties-certificate-revoked = Certificate: Revoked\npdfjs-digital-signature-properties-view-certificate = View certificate\npdfjs-digital-signature-properties-reason = Reason: { $reason }\npdfjs-digital-signature-properties-timestamp = Timestamp: { DATETIME($dateObj, dateStyle: \"short\", timeStyle: \"medium\") }\npdfjs-digital-signature-properties-sub-signatures =\n    { $count ->\n        [one] Sub-signature ({ $count })\n       *[other] Sub-signatures ({ $count })\n    }";
    return createBundle(lang, text);
  }
}

;// ./web/pdf_history.js





const HASH_CHANGE_TIMEOUT = 1000;
const POSITION_UPDATED_THRESHOLD = 50;
const UPDATE_VIEWAREA_TIMEOUT = 1000;
function getCurrentHash() {
  return document.location.hash;
}
class PDFHistory {
  #eventAC = null;
  constructor({
    linkService,
    eventBus
  }) {
    this.linkService = linkService;
    this.eventBus = eventBus;
    this._initialized = false;
    this._fingerprint = "";
    this.reset();
    this.eventBus.on("pagesinit", () => {
      this._isPagesLoaded = false;
      this.eventBus.on("pagesloaded", evt => {
        this._isPagesLoaded = !!evt.pagesCount;
      }, {
        once: true,
        ...internalOpt
      });
    }, internalOpt);
  }
  initialize({
    fingerprint,
    resetHistory = false,
    updateUrl = false
  }) {
    if (!fingerprint || typeof fingerprint !== "string") {
      console.error('PDFHistory.initialize: The "fingerprint" must be a non-empty string.');
      return;
    }
    if (this._initialized) {
      this.reset();
    }
    const reInitialized = this._fingerprint !== "" && this._fingerprint !== fingerprint;
    this._fingerprint = fingerprint;
    this._updateUrl = updateUrl === true;
    this._initialized = true;
    this.#bindEvents();
    const state = window.history.state;
    this._popStateInProgress = false;
    this._blockHashChange = 0;
    this._currentHash = getCurrentHash();
    this._numPositionUpdates = 0;
    this._uid = this._maxUid = 0;
    this._destination = null;
    this._position = null;
    if (!this.#isValidState(state, true) || resetHistory) {
      const {
        hash,
        page,
        rotation
      } = this.#parseCurrentHash(true);
      if (!hash || reInitialized || resetHistory) {
        this.#pushOrReplaceState(null, true);
        return;
      }
      this.#pushOrReplaceState({
        hash,
        page,
        rotation
      }, true);
      return;
    }
    const destination = state.destination;
    this.#updateInternalState(destination, state.uid, true);
    if (destination.rotation !== undefined) {
      this._initialRotation = destination.rotation;
    }
    if (destination.dest) {
      this._initialBookmark = JSON.stringify(destination.dest);
      this._destination.page = null;
    } else if (destination.hash) {
      this._initialBookmark = destination.hash;
    } else if (destination.page) {
      this._initialBookmark = `page=${destination.page}`;
    }
  }
  reset() {
    if (this._initialized) {
      this.#pageHide();
      this._initialized = false;
      this.#unbindEvents();
    }
    if (this._updateViewareaTimeout) {
      clearTimeout(this._updateViewareaTimeout);
      this._updateViewareaTimeout = null;
    }
    this._initialBookmark = null;
    this._initialRotation = null;
  }
  push({
    namedDest = null,
    explicitDest,
    pageNumber
  }) {
    if (!this._initialized) {
      return;
    }
    if (namedDest && typeof namedDest !== "string") {
      console.error("PDFHistory.push: " + `"${namedDest}" is not a valid namedDest parameter.`);
      return;
    } else if (!Array.isArray(explicitDest)) {
      console.error("PDFHistory.push: " + `"${explicitDest}" is not a valid explicitDest parameter.`);
      return;
    } else if (!this.#isValidPage(pageNumber)) {
      if (pageNumber !== null || this._destination) {
        console.error("PDFHistory.push: " + `"${pageNumber}" is not a valid pageNumber parameter.`);
        return;
      }
    }
    const hash = namedDest || JSON.stringify(explicitDest);
    if (!hash) {
      return;
    }
    let forceReplace = false;
    if (this._destination && (isDestHashesEqual(this._destination.hash, hash) || isDestArraysEqual(this._destination.dest, explicitDest))) {
      if (this._destination.page) {
        return;
      }
      forceReplace = true;
    }
    if (this._popStateInProgress && !forceReplace) {
      return;
    }
    this.#pushOrReplaceState({
      dest: explicitDest,
      hash,
      page: pageNumber,
      rotation: this.linkService.rotation
    }, forceReplace);
    if (!this._popStateInProgress) {
      this._popStateInProgress = true;
      Promise.resolve().then(() => {
        this._popStateInProgress = false;
      });
    }
  }
  pushPage(pageNumber) {
    if (!this._initialized) {
      return;
    }
    if (!this.#isValidPage(pageNumber)) {
      console.error(`PDFHistory.pushPage: "${pageNumber}" is not a valid page number.`);
      return;
    }
    if (this._destination?.page === pageNumber) {
      return;
    }
    if (this._popStateInProgress) {
      return;
    }
    this.#pushOrReplaceState({
      dest: null,
      hash: `page=${pageNumber}`,
      page: pageNumber,
      rotation: this.linkService.rotation
    });
    if (!this._popStateInProgress) {
      this._popStateInProgress = true;
      Promise.resolve().then(() => {
        this._popStateInProgress = false;
      });
    }
  }
  pushCurrentPosition() {
    if (!this._initialized || this._popStateInProgress) {
      return;
    }
    this.#tryPushCurrentPosition();
  }
  back() {
    if (!this._initialized || this._popStateInProgress) {
      return;
    }
    const state = window.history.state;
    if (this.#isValidState(state) && state.uid > 0) {
      window.history.back();
    }
  }
  forward() {
    if (!this._initialized || this._popStateInProgress) {
      return;
    }
    const state = window.history.state;
    if (this.#isValidState(state) && state.uid < this._maxUid) {
      window.history.forward();
    }
  }
  get popStateInProgress() {
    return this._initialized && (this._popStateInProgress || this._blockHashChange > 0);
  }
  get initialBookmark() {
    return this._initialized ? this._initialBookmark : null;
  }
  get initialRotation() {
    return this._initialized ? this._initialRotation : null;
  }
  #pushOrReplaceState(destination, forceReplace = false) {
    const shouldReplace = forceReplace || !this._destination;
    const newState = {
      fingerprint: this._fingerprint,
      uid: shouldReplace ? this._uid : this._uid + 1,
      destination
    };
    this.#updateInternalState(destination, newState.uid);
    let newUrl;
    if (this._updateUrl && destination?.hash) {
      const {
        href,
        protocol
      } = document.location;
      if (protocol !== "file:") {
        newUrl = updateUrlHash(href, destination.hash);
      }
    }
    if (shouldReplace) {
      window.history.replaceState(newState, "", newUrl);
    } else {
      window.history.pushState(newState, "", newUrl);
    }
  }
  #tryPushCurrentPosition(temporary = false) {
    if (!this._position) {
      return;
    }
    let position = this._position;
    if (temporary) {
      position = Object.assign(Object.create(null), this._position);
      position.temporary = true;
    }
    if (!this._destination) {
      this.#pushOrReplaceState(position);
      return;
    }
    if (this._destination.temporary) {
      this.#pushOrReplaceState(position, true);
      return;
    }
    if (this._destination.hash === position.hash) {
      return;
    }
    if (!this._destination.page && ( false || this._numPositionUpdates <= POSITION_UPDATED_THRESHOLD)) {
      return;
    }
    let forceReplace = false;
    if (this._destination.page >= position.first && this._destination.page <= position.page) {
      if (this._destination.dest !== undefined || !this._destination.first) {
        return;
      }
      forceReplace = true;
    }
    this.#pushOrReplaceState(position, forceReplace);
  }
  #isValidPage(val) {
    return Number.isInteger(val) && val > 0 && val <= this.linkService.pagesCount;
  }
  #isValidState(state, checkReload = false) {
    if (!state) {
      return false;
    }
    if (state.fingerprint !== this._fingerprint) {
      if (checkReload) {
        if (typeof state.fingerprint !== "string" || state.fingerprint.length !== this._fingerprint.length) {
          return false;
        }
        const [perfEntry] = performance.getEntriesByType("navigation");
        if (perfEntry?.type !== "reload") {
          return false;
        }
      } else {
        return false;
      }
    }
    if (!Number.isInteger(state.uid) || state.uid < 0) {
      return false;
    }
    if (state.destination === null || typeof state.destination !== "object") {
      return false;
    }
    return true;
  }
  #updateInternalState(destination, uid, removeTemporary = false) {
    if (this._updateViewareaTimeout) {
      clearTimeout(this._updateViewareaTimeout);
      this._updateViewareaTimeout = null;
    }
    if (removeTemporary && destination?.temporary) {
      delete destination.temporary;
    }
    this._destination = destination;
    this._uid = uid;
    this._maxUid = Math.max(this._maxUid, uid);
    this._numPositionUpdates = 0;
  }
  #parseCurrentHash(checkNameddest = false) {
    const hash = unescape(getCurrentHash()).substring(1);
    const params = parseQueryString(hash);
    const nameddest = params.get("nameddest") || "";
    let page = params.get("page") | 0;
    if (!this.#isValidPage(page) || checkNameddest && nameddest.length > 0) {
      page = null;
    }
    return {
      hash,
      page,
      rotation: this.linkService.rotation
    };
  }
  #updateViewarea({
    location
  }) {
    if (this._updateViewareaTimeout) {
      clearTimeout(this._updateViewareaTimeout);
      this._updateViewareaTimeout = null;
    }
    this._position = {
      hash: location.pdfOpenParams.substring(1),
      page: this.linkService.page,
      first: location.pageNumber,
      rotation: location.rotation
    };
    if (this._popStateInProgress) {
      return;
    }
    if ( true && this._isPagesLoaded && this._destination && !this._destination.page) {
      this._numPositionUpdates++;
    }
    if (true) {
      this._updateViewareaTimeout = setTimeout(() => {
        if (!this._popStateInProgress) {
          this.#tryPushCurrentPosition(true);
        }
        this._updateViewareaTimeout = null;
      }, UPDATE_VIEWAREA_TIMEOUT);
    }
  }
  #popState({
    state
  }) {
    const newHash = getCurrentHash(),
      hashChanged = this._currentHash !== newHash;
    this._currentHash = newHash;
    if (!state) {
      this._uid++;
      const {
        hash,
        page,
        rotation
      } = this.#parseCurrentHash();
      this.#pushOrReplaceState({
        hash,
        page,
        rotation
      }, true);
      return;
    }
    if (!this.#isValidState(state)) {
      return;
    }
    this._popStateInProgress = true;
    if (hashChanged) {
      this._blockHashChange++;
      waitOnEventOrTimeout({
        target: window,
        name: "hashchange",
        delay: HASH_CHANGE_TIMEOUT
      }).then(() => {
        this._blockHashChange--;
      });
    }
    const destination = state.destination;
    this.#updateInternalState(destination, state.uid, true);
    if (isValidRotation(destination.rotation)) {
      this.linkService.rotation = destination.rotation;
    }
    if (destination.dest) {
      this.linkService.goToDestination(destination.dest);
    } else if (destination.hash) {
      this.linkService.setHash(destination.hash);
    } else if (destination.page) {
      this.linkService.page = destination.page;
    }
    Promise.resolve().then(() => {
      this._popStateInProgress = false;
    });
  }
  #pageHide() {
    if (!this._destination || this._destination.temporary) {
      this.#tryPushCurrentPosition();
    }
  }
  #bindEvents() {
    if (this.#eventAC) {
      return;
    }
    this.#eventAC = new AbortController();
    const {
      signal
    } = this.#eventAC;
    this.eventBus.on("updateviewarea", this.#updateViewarea.bind(this), {
      signal,
      ...internalOpt
    });
    window.addEventListener("popstate", this.#popState.bind(this), {
      signal
    });
    window.addEventListener("pagehide", this.#pageHide.bind(this), {
      signal
    });
  }
  #unbindEvents() {
    this.#eventAC?.abort();
    this.#eventAC = null;
  }
}
function isDestHashesEqual(destHash, pushHash) {
  if (typeof destHash !== "string" || typeof pushHash !== "string") {
    return false;
  }
  if (destHash === pushHash) {
    return true;
  }
  const nameddest = parseQueryString(destHash).get("nameddest");
  if (nameddest === pushHash) {
    return true;
  }
  return false;
}
function isDestArraysEqual(firstDest, secondDest) {
  function isEntryEqual(first, second) {
    if (typeof first !== typeof second) {
      return false;
    }
    if (Array.isArray(first) || Array.isArray(second)) {
      return false;
    }
    if (first !== null && typeof first === "object" && second !== null) {
      if (Object.keys(first).length !== Object.keys(second).length) {
        return false;
      }
      for (const key in first) {
        if (!isEntryEqual(first[key], second[key])) {
          return false;
        }
      }
      return true;
    }
    return first === second || Number.isNaN(first) && Number.isNaN(second);
  }
  if (!(Array.isArray(firstDest) && Array.isArray(secondDest))) {
    return false;
  }
  if (firstDest.length !== secondDest.length) {
    return false;
  }
  for (let i = 0, ii = firstDest.length; i < ii; i++) {
    if (!isEntryEqual(firstDest[i], secondDest[i])) {
      return false;
    }
  }
  return true;
}

;// ./web/annotation_editor_layer_builder.js


class AnnotationEditorLayerBuilder {
  #annotationLayer = null;
  #drawLayer = null;
  #onAppend = null;
  #structTreeLayer = null;
  #textLayer = null;
  #uiManager;
  constructor(options) {
    this.pageIndex = options.pageIndex;
    this.accessibilityManager = options.accessibilityManager;
    this.l10n = options.l10n;
    this.l10n ||= new genericl10n_GenericL10n();
    this.annotationEditorLayer = null;
    this.div = null;
    this._cancelled = false;
    this.#uiManager = options.uiManager;
    this.#annotationLayer = options.annotationLayer || null;
    this.#textLayer = options.textLayer || null;
    this.#drawLayer = options.drawLayer || null;
    this.#onAppend = options.onAppend || null;
    this.#structTreeLayer = options.structTreeLayer || null;
  }
  updatePageIndex(newPageIndex) {
    this.pageIndex = newPageIndex;
    this.annotationEditorLayer?.updatePageIndex(newPageIndex);
  }
  async render({
    viewport,
    intent = "display"
  }) {
    if (intent !== "display") {
      return;
    }
    if (this._cancelled) {
      return;
    }
    const clonedViewport = viewport.clone({
      dontFlip: true
    });
    if (this.div) {
      this.annotationEditorLayer.update({
        viewport: clonedViewport
      });
      this.show();
      return;
    }
    const div = this.div = document.createElement("div");
    div.className = "annotationEditorLayer";
    div.hidden = true;
    div.dir = this.#uiManager.direction;
    this.#onAppend?.(div);
    this.annotationEditorLayer = new AnnotationEditorLayer({
      uiManager: this.#uiManager,
      div,
      structTreeLayer: this.#structTreeLayer,
      accessibilityManager: this.accessibilityManager,
      pageIndex: this.pageIndex,
      l10n: this.l10n,
      viewport: clonedViewport,
      annotationLayer: this.#annotationLayer,
      textLayer: this.#textLayer,
      drawLayer: this.#drawLayer
    });
    const parameters = {
      viewport: clonedViewport,
      div,
      annotations: null,
      intent
    };
    await this.annotationEditorLayer.render(parameters);
    this.show();
  }
  update(viewport) {
    if (this.div) {
      this.annotationEditorLayer.update({
        viewport: viewport.clone({
          dontFlip: true
        })
      });
    }
  }
  cancel() {
    this._cancelled = true;
    if (!this.div) {
      return;
    }
    this.annotationEditorLayer.destroy();
  }
  hide() {
    if (!this.div) {
      return;
    }
    this.annotationEditorLayer.pause(true);
    this.div.hidden = true;
  }
  show() {
    if (!this.div || this.annotationEditorLayer.isInvisible) {
      return;
    }
    this.div.hidden = false;
    this.annotationEditorLayer.pause(false);
  }
}

;// ./web/app_options.js




{
  var compatParams = new Map();
  const {
    maxTouchPoints,
    platform,
    userAgent
  } = navigator;
  const isAndroid = /Android/.test(userAgent);
  const isIOS = /\b(?:iPad|iPhone|iPod)(?=;)/.test(userAgent) || platform === "MacIntel" && maxTouchPoints > 1;
  if (isIOS || isAndroid) {
    compatParams.set("maxCanvasPixels", 5242880);
  }
  if (isAndroid) {
    compatParams.set("useSystemFonts", false);
  }
}
const OptionKind = {
  BROWSER: 0x01,
  VIEWER: 0x02,
  API: 0x04,
  WORKER: 0x08,
  EVENT_DISPATCH: 0x10,
  PREFERENCE: 0x80
};
const Type = {
  BOOLEAN: 0x01,
  NUMBER: 0x02,
  OBJECT: 0x04,
  STRING: 0x08,
  UNDEFINED: 0x10
};
const defaultOptions = new Map([["allowedGlobalEvents", {
  value: null,
  kind: OptionKind.BROWSER
}], ["canvasMaxAreaInBytes", {
  value: -1,
  kind: OptionKind.BROWSER + OptionKind.API
}], ["isInAutomation", {
  value: false,
  kind: OptionKind.BROWSER
}], ["localeProperties", {
  value: {
    lang: navigator.language || "en-US"
  },
  kind: OptionKind.BROWSER
}], ["maxCanvasDim", {
  value: 32767,
  kind: OptionKind.BROWSER + OptionKind.VIEWER
}], ["nimbusDataStr", {
  value: "",
  kind: OptionKind.BROWSER
}], ["supportsCaretBrowsingMode", {
  value: false,
  kind: OptionKind.BROWSER
}], ["supportsDocumentFonts", {
  value: true,
  kind: OptionKind.BROWSER
}], ["supportsDownloading", {
  value: true,
  kind: OptionKind.BROWSER
}], ["supportsIntegratedFind", {
  value: false,
  kind: OptionKind.BROWSER
}], ["supportsMouseWheelZoomCtrlKey", {
  value: true,
  kind: OptionKind.BROWSER
}], ["supportsMouseWheelZoomMetaKey", {
  value: true,
  kind: OptionKind.BROWSER
}], ["supportsPinchToZoom", {
  value: true,
  kind: OptionKind.BROWSER
}], ["supportsPrinting", {
  value: true,
  kind: OptionKind.BROWSER
}], ["toolbarDensity", {
  value: 0,
  kind: OptionKind.BROWSER + OptionKind.EVENT_DISPATCH
}], ["altTextLearnMoreUrl", {
  value: "",
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["annotationEditorMode", {
  value: 0,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["annotationMode", {
  value: 2,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["capCanvasAreaFactor", {
  value: 200,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["commentLearnMoreUrl", {
  value: "",
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["cursorToolOnLoad", {
  value: 0,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["debuggerSrc", {
  value: "./debugger.mjs",
  kind: OptionKind.VIEWER
}], ...[["defaultUrl", {
  value: "compressed.tracemonkey-pldi-09.pdf",
  kind: OptionKind.VIEWER
}]], ["defaultZoomDelay", {
  value: 400,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["defaultZoomValue", {
  value: "",
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["disableHistory", {
  value: false,
  kind: OptionKind.VIEWER
}], ["disablePageLabels", {
  value: false,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ...[["disablePreferences", {
  value: false,
  kind: OptionKind.VIEWER
}]], ...[], ["enableAltText", {
  value: false,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["enableAltTextModelDownload", {
  value: true,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE + OptionKind.EVENT_DISPATCH
}], ["enableAutoLinking", {
  value: true,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["enableComment", {
  value: false,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["enableDetailCanvas", {
  value: true,
  kind: OptionKind.VIEWER
}], ...[], ["enableGuessAltText", {
  value: true,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE + OptionKind.EVENT_DISPATCH
}], ["enableHighlightFloatingButton", {
  value: false,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["enableMerge", {
  value: false,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["enableNewAltTextWhenAddingImage", {
  value: true,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["enableNewBadge", {
  value: false,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["enableNova", {
  value: true,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["enableOptimizedPartialRendering", {
  value: false,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["enablePermissions", {
  value: false,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["enablePrintAutoRotate", {
  value: true,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["enableScripting", {
  value: true,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["enableSelectionRendering", {
  value: true,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["enableSignatureEditor", {
  value: false,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["enableSignatureVerification", {
  value: false,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["enableSplitMerge", {
  value: false,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["enableUpdatedAddImage", {
  value: false,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["externalLinkRel", {
  value: "noopener noreferrer nofollow",
  kind: OptionKind.VIEWER
}], ["externalLinkTarget", {
  value: 0,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ...[], ["highlightEditorColors", {
  value: "yellow=#FFFF98,green=#53FFBC,blue=#80EBFF,pink=#FFCBE6,red=#FF4F5F," + "yellow_HCM=#FFFFCC,green_HCM=#53FFBC,blue_HCM=#80EBFF,pink_HCM=#F6B8FF,red_HCM=#C50043",
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["historyUpdateUrl", {
  value: false,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["ignoreDestinationZoom", {
  value: false,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["imageResourcesPath", {
  value: "./images/",
  kind: OptionKind.VIEWER
}], ["imagesRightClickMinSize", {
  value: -1,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["maxCanvasPixels", {
  value: 2 ** 25,
  kind: OptionKind.VIEWER
}], ["minDurationToUpdateCanvas", {
  value: 500,
  kind: OptionKind.VIEWER
}], ["forcePageColors", {
  value: false,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["pageColorsBackground", {
  value: "Canvas",
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["pageColorsForeground", {
  value: "CanvasText",
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["pdfBugEnabled", {
  value: false,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["postMessageAfterPrintCallback", {
  value: false,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["printResolution", {
  value: 150,
  kind: OptionKind.VIEWER
}], ...[["sandboxBundleSrc", {
  value: "../build/pdf.sandbox.mjs",
  kind: OptionKind.VIEWER
}]], ["sidebarViewOnLoad", {
  value: -1,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["scrollModeOnLoad", {
  value: -1,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["spreadModeOnLoad", {
  value: -1,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["textLayerMode", {
  value: 1,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["viewerCssTheme", {
  value: 0,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["viewOnLoad", {
  value: 0,
  kind: OptionKind.VIEWER + OptionKind.PREFERENCE
}], ["cMapPacked", {
  value: true,
  kind: OptionKind.API
}], ["cMapUrl", {
  value: "../web/cmaps/",
  kind: OptionKind.API
}], ["disableAutoFetch", {
  value: false,
  kind: OptionKind.API + OptionKind.PREFERENCE
}], ["disableFontFace", {
  value: false,
  kind: OptionKind.API + OptionKind.PREFERENCE
}], ["disableRange", {
  value: false,
  kind: OptionKind.API + OptionKind.PREFERENCE
}], ["disableStream", {
  value: false,
  kind: OptionKind.API + OptionKind.PREFERENCE
}], ["docBaseUrl", {
  value: "",
  kind: OptionKind.API
}], ["enableHWA", {
  value: true,
  kind: OptionKind.API + OptionKind.PREFERENCE
}], ["enableWebGPU", {
  value: true,
  kind: OptionKind.API + OptionKind.PREFERENCE
}], ["enableXfa", {
  value: true,
  kind: OptionKind.API + OptionKind.PREFERENCE
}], ["fontExtraProperties", {
  value: false,
  kind: OptionKind.API
}], ["iccUrl", {
  value: "../web/iccs/",
  kind: OptionKind.API
}], ["isOffscreenCanvasSupported", {
  value: true,
  kind: OptionKind.API
}], ["maxImageSize", {
  value: -1,
  kind: OptionKind.API
}], ["pdfBug", {
  value: false,
  kind: OptionKind.API
}], ["standardFontDataUrl", {
  value: "../web/standard_fonts/",
  kind: OptionKind.API
}], ["useSystemFonts", {
  value: undefined,
  kind: OptionKind.API,
  type: Type.BOOLEAN + Type.UNDEFINED
}], ["verbosity", {
  value: 1,
  kind: OptionKind.API
}], ["wasmUrl", {
  value: "../web/wasm/",
  kind: OptionKind.API
}], ["workerPort", {
  value: null,
  kind: OptionKind.WORKER
}], ["workerSrc", {
  value: "../build/pdf.worker.mjs",
  kind: OptionKind.WORKER
}]]);
class AppOptions {
  static eventBus;
  static #opts = new Map();
  static {
    for (const [name, defaultOpt] of defaultOptions) {
      this.#opts.set(name, defaultOpt.value);
    }
    for (const [name, value] of compatParams) {
      this.#opts.set(name, value);
    }
    this._hasInvokedSet = false;
    this._checkDisablePreferences = () => {
      if (this.get("disablePreferences")) {
        return true;
      }
      if (this._hasInvokedSet) {
        console.warn("The Preferences may override manually set AppOptions; " + 'please use the "disablePreferences"-option to prevent that.');
      }
      return false;
    };
  }
  static get(name) {
    return this.#opts.get(name);
  }
  static getAll(kind = null, defaultOnly = false) {
    const options = Object.create(null);
    for (const [name, defaultOpt] of defaultOptions) {
      if (kind && !(kind & defaultOpt.kind)) {
        continue;
      }
      options[name] = !defaultOnly ? this.#opts.get(name) : defaultOpt.value;
    }
    return options;
  }
  static set(name, value) {
    this.setAll({
      [name]: value
    });
  }
  static setAll(options, prefs = false) {
    this._hasInvokedSet ||= true;
    let events;
    for (const name in options) {
      const defaultOpt = defaultOptions.get(name),
        userOpt = options[name];
      if (!defaultOpt || !(typeof userOpt === typeof defaultOpt.value || Type[(typeof userOpt).toUpperCase()] & defaultOpt.type)) {
        continue;
      }
      const {
        kind
      } = defaultOpt;
      if (prefs && !(kind & OptionKind.BROWSER || kind & OptionKind.PREFERENCE)) {
        continue;
      }
      if (this.eventBus && kind & OptionKind.EVENT_DISPATCH) {
        (events ??= new Map()).set(name, userOpt);
      }
      this.#opts.set(name, userOpt);
    }
    events?.forEach((value, name) => {
      this.eventBus.dispatch(name.toLowerCase(), {
        source: this,
        value
      });
    });
  }
}

;// ./web/autolinker.js






function DOMRectToPDF({
  width,
  height,
  left,
  top
}, pdfPageView) {
  if (width === 0 || height === 0) {
    return null;
  }
  const pageBox = pdfPageView.textLayer.div.getBoundingClientRect();
  const bottomLeft = pdfPageView.getPagePoint(left - pageBox.left, top - pageBox.top);
  const topRight = pdfPageView.getPagePoint(left - pageBox.left + width, top - pageBox.top + height);
  return Util.normalizeRect([bottomLeft[0], bottomLeft[1], topRight[0], topRight[1]]);
}
function calculateLinkPosition(range, pdfPageView) {
  const rangeRects = range.getClientRects();
  if (rangeRects.length === 1) {
    return {
      rect: DOMRectToPDF(rangeRects[0], pdfPageView)
    };
  }
  const rect = [Infinity, Infinity, -Infinity, -Infinity];
  const quadPoints = [];
  let i = 0;
  for (const domRect of rangeRects) {
    const normalized = DOMRectToPDF(domRect, pdfPageView);
    if (normalized === null) {
      continue;
    }
    quadPoints[i] = quadPoints[i + 4] = normalized[0];
    quadPoints[i + 1] = quadPoints[i + 3] = normalized[3];
    quadPoints[i + 2] = quadPoints[i + 6] = normalized[2];
    quadPoints[i + 5] = quadPoints[i + 7] = normalized[1];
    Util.rectBoundingBox(...normalized, rect);
    i += 8;
  }
  return {
    quadPoints,
    rect
  };
}
function textPosition(container, offset) {
  let currentContainer = container;
  do {
    if (currentContainer.nodeType === Node.TEXT_NODE) {
      const currentLength = currentContainer.textContent.length;
      if (offset <= currentLength) {
        return [currentContainer, offset];
      }
      offset -= currentLength;
    } else if (currentContainer.firstChild) {
      currentContainer = currentContainer.firstChild;
      continue;
    }
    while (!currentContainer.nextSibling && currentContainer !== container) {
      currentContainer = currentContainer.parentNode;
    }
    if (currentContainer !== container) {
      currentContainer = currentContainer.nextSibling;
    }
  } while (currentContainer !== container);
  throw new Error("Offset is bigger than container's contents length.");
}
function createLinkAnnotation({
  url,
  index,
  length
}, pdfPageView, id) {
  const highlighter = pdfPageView._textHighlighter;
  const [{
    begin,
    end
  }] = highlighter._convertMatches([index], [length]);
  const range = new Range();
  range.setStart(...textPosition(highlighter.textDivs[begin.divIdx], begin.offset));
  range.setEnd(...textPosition(highlighter.textDivs[end.divIdx], end.offset));
  return {
    id: `inferred_link_${id}`,
    unsafeUrl: url,
    url,
    annotationType: AnnotationType.LINK,
    rotation: 0,
    ...calculateLinkPosition(range, pdfPageView),
    borderStyle: null
  };
}
class Autolinker {
  static #index = 0;
  static #regex;
  static #numericTLDRegex;
  static findLinks(text) {
    this.#regex ??= /\b(?:https?:\/\/|mailto:|www\.)(?:[\S--[\p{P}<>]]|\/|[\S--[\[\]]]+[\S--[\p{P}<>]])+|(?=\p{L})[\S--[@\p{Ps}\p{Pe}<>]]{1,64}@([\S--[[\p{P}--\-]<>]]{1,63}(?:\.[\S--[[\p{P}--\-]<>]]{1,63})+)/gv;
    const [normalizedText, diffs] = normalize(text, {
      ignoreDashEOL: true
    });
    const matches = normalizedText.matchAll(this.#regex);
    const links = [];
    for (const match of matches) {
      const [url, emailDomain] = match;
      let raw;
      if (url.startsWith("www.") || url.startsWith("http://") || url.startsWith("https://")) {
        raw = url;
      } else if (emailDomain) {
        const hostname = URL.parse(`http://${emailDomain}`)?.hostname;
        if (!hostname) {
          continue;
        }
        this.#numericTLDRegex ??= /\.\d+$/;
        if (this.#numericTLDRegex.test(hostname)) {
          continue;
        }
      }
      raw ??= url.startsWith("mailto:") ? url : `mailto:${url}`;
      const absoluteURL = createValidAbsoluteUrl(raw, null, {
        addDefaultProtocol: true
      });
      if (absoluteURL) {
        const [index, length] = getOriginalIndex(diffs, match.index, url.length);
        links.push({
          url: absoluteURL.href,
          index,
          length
        });
      }
    }
    return links;
  }
  static processLinks(pdfPageView) {
    return this.findLinks(pdfPageView._textHighlighter.textContentItemsStr.join("\n")).map(link => createLinkAnnotation(link, pdfPageView, this.#index++));
  }
}

;// ./web/renderable_view.js
const RenderingStates = {
  INITIAL: 0,
  RUNNING: 1,
  PAUSED: 2,
  FINISHED: 3
};
class RenderableView {
  renderingId = "";
  renderTask = null;
  resume = null;
  get renderingState() {
    throw new Error("Abstract getter `renderingState` accessed");
  }
  set renderingState(state) {
    throw new Error("Abstract setter `renderingState` accessed");
  }
  async draw() {
    throw new Error("Not implemented: draw");
  }
}

;// ./web/base_pdf_page_view.js


class BasePDFPageView extends RenderableView {
  #loadingId = null;
  #renderError = null;
  #renderingState = RenderingStates.INITIAL;
  #showCanvas = null;
  #startTime = 0;
  #tempCanvas = null;
  canvas = null;
  div = null;
  enableOptimizedPartialRendering = false;
  enableSelectionRendering = true;
  imagesRightClickMinSize = -1;
  eventBus = null;
  id = null;
  imageCoordinates = null;
  pageColors = null;
  recordedBBoxes = null;
  renderingQueue = null;
  constructor(options) {
    super();
    this.eventBus = options.eventBus;
    this.id = options.id;
    this.pageColors = options.pageColors || null;
    this.renderingQueue = options.renderingQueue;
    this.enableOptimizedPartialRendering = options.enableOptimizedPartialRendering ?? false;
    this.enableSelectionRendering = options.enableSelectionRendering !== false && FeatureTest.isBackdropFilterSupported;
    this.imagesRightClickMinSize = options.imagesRightClickMinSize ?? -1;
    this.minDurationToUpdateCanvas = options.minDurationToUpdateCanvas ?? 500;
  }
  get renderingState() {
    return this.#renderingState;
  }
  set renderingState(state) {
    if (state === this.#renderingState) {
      return;
    }
    this.#renderingState = state;
    if (this.#loadingId) {
      clearTimeout(this.#loadingId);
      this.#loadingId = null;
    }
    switch (state) {
      case RenderingStates.PAUSED:
        this.div.classList.remove("loading");
        this.#startTime = 0;
        this.#showCanvas?.(false);
        break;
      case RenderingStates.RUNNING:
        this.div.classList.add("loadingIcon");
        this.#loadingId = setTimeout(() => {
          this.div.classList.add("loading");
          this.#loadingId = null;
        }, 0);
        this.#startTime = Date.now();
        break;
      case RenderingStates.INITIAL:
      case RenderingStates.FINISHED:
        this.div.classList.remove("loadingIcon", "loading");
        this.#startTime = 0;
        break;
    }
  }
  _createCanvas(onShow, hideUntilComplete = false) {
    const {
      pageColors
    } = this;
    const hasHCM = !!(pageColors?.background && pageColors?.foreground);
    const prevCanvas = this.canvas;
    const updateOnFirstShow = !prevCanvas && !hasHCM && !hideUntilComplete;
    let canvas = this.canvas = document.createElement("canvas");
    this.#showCanvas = isLastShow => {
      if (updateOnFirstShow) {
        let tempCanvas = this.#tempCanvas;
        if (!isLastShow && this.minDurationToUpdateCanvas > 0) {
          if (Date.now() - this.#startTime < this.minDurationToUpdateCanvas) {
            return;
          }
          if (!tempCanvas) {
            tempCanvas = this.#tempCanvas = canvas;
            canvas = this.canvas = canvas.cloneNode(false);
            onShow(canvas);
          }
        }
        if (tempCanvas) {
          const ctx = canvas.getContext("2d", {
            alpha: false
          });
          ctx.drawImage(tempCanvas, 0, 0);
          if (isLastShow) {
            this.#resetTempCanvas();
          } else {
            this.#startTime = Date.now();
          }
          return;
        }
        onShow(canvas);
        this.#showCanvas = null;
        return;
      }
      if (!isLastShow) {
        return;
      }
      if (prevCanvas) {
        prevCanvas.replaceWith(canvas);
        prevCanvas.width = prevCanvas.height = 0;
      } else {
        onShow(canvas);
      }
    };
    return {
      canvas,
      prevCanvas
    };
  }
  #renderContinueCallback = cont => {
    this.#showCanvas?.(false);
    if (this.renderingQueue && !this.renderingQueue.isHighestPriority(this)) {
      this.renderingState = RenderingStates.PAUSED;
      this.resume = () => {
        this.renderingState = RenderingStates.RUNNING;
        cont();
      };
      return;
    }
    cont();
  };
  _resetCanvas() {
    const {
      canvas
    } = this;
    if (!canvas) {
      return;
    }
    canvas.remove();
    canvas.width = canvas.height = 0;
    this.canvas = null;
    this.#resetTempCanvas();
  }
  #resetTempCanvas() {
    if (this.#tempCanvas) {
      this.#tempCanvas.width = this.#tempCanvas.height = 0;
      this.#tempCanvas = null;
    }
  }
  async _drawCanvas(options, onCancel, onFinish) {
    const renderTask = this.renderTask = this.pdfPage.render(options);
    renderTask.onContinue = this.#renderContinueCallback;
    renderTask.onError = error => {
      if (error instanceof RenderingCancelledException) {
        onCancel();
        this.#renderError = null;
      }
    };
    let error = null;
    try {
      await renderTask.promise;
      this.#showCanvas?.(true);
    } catch (e) {
      if (e instanceof RenderingCancelledException) {
        return;
      }
      error = e;
      this.#showCanvas?.(true);
    } finally {
      this.#renderError = error;
      if (renderTask === this.renderTask) {
        this.renderTask = null;
        if (this.enableOptimizedPartialRendering) {
          this.recordedBBoxes ??= renderTask.recordedBBoxes;
        }
        if (this.imagesRightClickMinSize !== -1) {
          this.imageCoordinates ??= this.pdfPage.imageCoordinates;
        }
      }
    }
    this.renderingState = RenderingStates.FINISHED;
    onFinish(renderTask);
    if (error) {
      throw error;
    }
  }
  cancelRendering({
    cancelExtraDelay = 0
  } = {}) {
    if (this.renderTask) {
      this.renderTask.cancel(cancelExtraDelay);
      this.renderTask = null;
    }
    this.resume = null;
  }
  dispatchPageRender() {
    this.eventBus.dispatch("pagerender", {
      source: this,
      pageNumber: this.id
    });
  }
  dispatchPageRendered(cssTransform, isDetailView) {
    this.eventBus.dispatch("pagerendered", {
      source: this,
      pageNumber: this.id,
      cssTransform,
      isDetailView,
      timestamp: performance.now(),
      error: this.#renderError
    });
  }
}

;// ./web/draw_layer_builder.js

class DrawLayerBuilder {
  #drawLayer = null;
  constructor(options) {
    this.pageIndex = options.pageIndex;
    this.textLayer = options.textLayer || null;
    this.filterFactory = options.filterFactory || null;
    this.pageColors = options.pageColors || null;
  }
  async render({
    intent = "display"
  }) {
    if (intent !== "display" || this.#drawLayer || this._cancelled) {
      return;
    }
    this.#drawLayer = new DrawLayer({
      pageIndex: this.pageIndex,
      textLayer: this.textLayer,
      filterFactory: this.filterFactory,
      pageColors: this.pageColors
    });
  }
  cancel() {
    this._cancelled = true;
    this.#drawLayer?.destroy();
    this.#drawLayer = null;
  }
  setParent(parent) {
    this.#drawLayer?.setParent(parent);
  }
  getDrawLayer() {
    return this.#drawLayer;
  }
}

;// ./web/pdf_page_detail_view.js



class PDFPageDetailView extends BasePDFPageView {
  #detailArea = null;
  renderingCancelled = false;
  constructor({
    pageView
  }) {
    super(pageView);
    this.pageView = pageView;
    this.renderingId = "detail" + this.id;
    this.div = pageView.div;
  }
  setPdfPage(pdfPage) {
    this.pageView.setPdfPage(pdfPage);
  }
  get pdfPage() {
    return this.pageView.pdfPage;
  }
  get renderingState() {
    return super.renderingState;
  }
  set renderingState(state) {
    this.renderingCancelled = false;
    super.renderingState = state;
  }
  reset({
    keepCanvas = false
  } = {}) {
    const renderingCancelled = this.renderingCancelled || this.renderingState === RenderingStates.RUNNING || this.renderingState === RenderingStates.PAUSED;
    this.cancelRendering();
    this.renderingState = RenderingStates.INITIAL;
    this.renderingCancelled = renderingCancelled;
    if (!keepCanvas) {
      this._resetCanvas();
    }
  }
  #shouldRenderDifferentArea(visibleArea) {
    if (!this.#detailArea) {
      return true;
    }
    const minDetailX = this.#detailArea.minX;
    const minDetailY = this.#detailArea.minY;
    const maxDetailX = this.#detailArea.width + minDetailX;
    const maxDetailY = this.#detailArea.height + minDetailY;
    if (visibleArea.minX < minDetailX || visibleArea.minY < minDetailY || visibleArea.maxX > maxDetailX || visibleArea.maxY > maxDetailY) {
      return true;
    }
    const {
      width: maxWidth,
      height: maxHeight,
      scale
    } = this.pageView.viewport;
    if (this.#detailArea.scale !== scale) {
      return true;
    }
    const paddingLeftSize = visibleArea.minX - minDetailX;
    const paddingRightSize = maxDetailX - visibleArea.maxX;
    const paddingTopSize = visibleArea.minY - minDetailY;
    const paddingBottomSize = maxDetailY - visibleArea.maxY;
    const MOVEMENT_THRESHOLD = 0.5;
    const ratio = (1 + MOVEMENT_THRESHOLD) / MOVEMENT_THRESHOLD;
    if (minDetailX > 0 && paddingRightSize / paddingLeftSize > ratio || maxDetailX < maxWidth && paddingLeftSize / paddingRightSize > ratio || minDetailY > 0 && paddingBottomSize / paddingTopSize > ratio || maxDetailY < maxHeight && paddingTopSize / paddingBottomSize > ratio) {
      return true;
    }
    return false;
  }
  update({
    visibleArea = null,
    underlyingViewUpdated = false
  } = {}) {
    if (underlyingViewUpdated) {
      this.cancelRendering();
      this.renderingState = RenderingStates.INITIAL;
      return;
    }
    if (!this.#shouldRenderDifferentArea(visibleArea)) {
      return;
    }
    const {
      viewport,
      maxCanvasPixels,
      capCanvasAreaFactor
    } = this.pageView;
    const visibleWidth = visibleArea.maxX - visibleArea.minX;
    const visibleHeight = visibleArea.maxY - visibleArea.minY;
    const visiblePixels = visibleWidth * visibleHeight * OutputScale.pixelRatio ** 2;
    const maxDetailToVisibleLinearRatio = Math.sqrt(OutputScale.capPixels(maxCanvasPixels, capCanvasAreaFactor) / visiblePixels);
    const maxOverflowScale = (maxDetailToVisibleLinearRatio - 1) / 2;
    let overflowScale = Math.min(1, maxOverflowScale);
    if (overflowScale < 0) {
      overflowScale = 0;
    }
    const overflowWidth = visibleWidth * overflowScale;
    const overflowHeight = visibleHeight * overflowScale;
    const minX = Math.max(0, visibleArea.minX - overflowWidth);
    const maxX = Math.min(viewport.width, visibleArea.maxX + overflowWidth);
    const minY = Math.max(0, visibleArea.minY - overflowHeight);
    const maxY = Math.min(viewport.height, visibleArea.maxY + overflowHeight);
    const width = maxX - minX;
    const height = maxY - minY;
    this.#detailArea = {
      minX,
      minY,
      width,
      height,
      scale: viewport.scale
    };
    this.reset({
      keepCanvas: true
    });
  }
  _getRenderingContext(canvas, transform) {
    const baseContext = this.pageView._getRenderingContext(canvas, transform, false);
    const recordedBBoxes = this.pdfPage.recordedBBoxes;
    if (!recordedBBoxes || !this.enableOptimizedPartialRendering) {
      return baseContext;
    }
    const {
      viewport: {
        width: vWidth,
        height: vHeight
      }
    } = this.pageView;
    const {
      width: aWidth,
      height: aHeight,
      minX: aMinX,
      minY: aMinY
    } = this.#detailArea;
    const detailMinX = aMinX / vWidth;
    const detailMinY = aMinY / vHeight;
    const detailMaxX = (aMinX + aWidth) / vWidth;
    const detailMaxY = (aMinY + aHeight) / vHeight;
    return {
      ...baseContext,
      operationsFilter(index) {
        if (recordedBBoxes.isEmpty(index)) {
          return false;
        }
        return recordedBBoxes.minX(index) <= detailMaxX && recordedBBoxes.maxX(index) >= detailMinX && recordedBBoxes.minY(index) <= detailMaxY && recordedBBoxes.maxY(index) >= detailMinY;
      }
    };
  }
  async draw() {
    if (this.pageView.detailView !== this) {
      return undefined;
    }
    const hideUntilComplete = this.pageView.renderingState === RenderingStates.FINISHED || this.renderingState === RenderingStates.FINISHED;
    if (this.renderingState !== RenderingStates.INITIAL) {
      console.error("Must be in new state before drawing");
      this.reset();
    }
    const {
      div,
      pdfPage,
      viewport
    } = this.pageView;
    if (!pdfPage) {
      this.renderingState = RenderingStates.FINISHED;
      throw new Error("pdfPage is not loaded");
    }
    this.renderingState = RenderingStates.RUNNING;
    const canvasWrapper = this.pageView._ensureCanvasWrapper();
    const {
      canvas,
      prevCanvas
    } = this._createCanvas(newCanvas => {
      if (canvasWrapper.firstElementChild?.tagName === "CANVAS") {
        canvasWrapper.firstElementChild.after(newCanvas);
      } else {
        canvasWrapper.prepend(newCanvas);
      }
    }, hideUntilComplete);
    canvas.ariaHidden = true;
    if (this.enableOptimizedPartialRendering) {
      canvas.className = "detailView";
    }
    const {
      width,
      height
    } = viewport;
    const area = this.#detailArea;
    const {
      pixelRatio
    } = OutputScale;
    const transform = [pixelRatio, 0, 0, pixelRatio, -area.minX * pixelRatio, -area.minY * pixelRatio];
    canvas.width = area.width * pixelRatio;
    canvas.height = area.height * pixelRatio;
    const {
      style
    } = canvas;
    style.width = `${area.width * 100 / width}%`;
    style.height = `${area.height * 100 / height}%`;
    style.top = `${area.minY * 100 / height}%`;
    style.left = `${area.minX * 100 / width}%`;
    const renderingPromise = this._drawCanvas(this._getRenderingContext(canvas, transform), () => {
      this.canvas?.remove();
      this.canvas = prevCanvas;
    }, () => {
      this.pageView._refreshAnnotationLayer();
      this.dispatchPageRendered(false, true);
    });
    div.setAttribute("data-loaded", true);
    this.dispatchPageRender();
    return renderingPromise;
  }
}

;// ./web/struct_tree_layer_builder.js














const PDF_ROLE_TO_HTML_ROLE = {
  Document: null,
  DocumentFragment: null,
  Part: "group",
  Art: "article",
  Sect: "group",
  Div: "group",
  BlockQuote: "blockquote",
  Aside: "note",
  NonStruct: "none",
  P: "paragraph",
  H: "heading",
  Title: null,
  FENote: "note",
  Sub: "group",
  Lbl: null,
  Span: null,
  Em: "emphasis",
  Strong: "strong",
  Note: "note",
  Code: "code",
  Link: "link",
  Annot: "note",
  Form: "form",
  Ruby: null,
  RB: null,
  RT: null,
  RP: null,
  Warichu: null,
  WT: null,
  WP: null,
  L: "list",
  LI: "listitem",
  LBody: null,
  Table: "table",
  TR: "row",
  TH: "columnheader",
  TD: "cell",
  THead: "rowgroup",
  TBody: "rowgroup",
  TFoot: "rowgroup",
  Caption: "caption",
  Figure: "figure",
  Formula: null,
  Artifact: null
};
const ARIA_ROLES_WITH_PROHIBITED_NAMES = new Set(["caption", "code", "emphasis", "generic", "none", "paragraph", "strong"]);
const MathMLElements = new Set(["math", "merror", "mfrac", "mi", "mmultiscripts", "mn", "mo", "mover", "mpadded", "mprescripts", "mroot", "mrow", "ms", "mspace", "msqrt", "mstyle", "msub", "msubsup", "msup", "mtable", "mtd", "mtext", "mtr", "munder", "munderover", "semantics"]);
const MathMLNamespace = "http://www.w3.org/1998/Math/MathML";
class MathMLSanitizer {
  static get sanitizer() {
    return shadow(this, "sanitizer", FeatureTest.isSanitizerSupported ? new Sanitizer({
      elements: Array.from(MathMLElements.keys(), name => ({
        name,
        namespace: MathMLNamespace
      })),
      replaceWithChildrenElements: [{
        name: "maction",
        namespace: MathMLNamespace
      }],
      attributes: ["dir", "displaystyle", "mathbackground", "mathcolor", "mathsize", "scriptlevel", "encoding", "display", "linethickness", "intent", "arg", "form", "fence", "separator", "lspace", "rspace", "stretchy", "symmetric", "maxsize", "minsize", "largeop", "movablelimits", "width", "height", "depth", "voffset", "accent", "accentunder", "columnspan", "rowspan"],
      comments: false
    }) : null);
  }
}
const HEADING_PATTERN = /^H(\d+)$/;
class StructTreeLayerBuilder {
  #promise;
  #treeDom = null;
  #treePromise;
  #annotationIds = new Set();
  #elementAttributes = new Map();
  #pendingLinkOwnership = new Map();
  #linkTextId = 0;
  #structElementIdPrefix = `pdfjs_internal_struct_${getUuid()}_`;
  #structElementIds = new Map();
  #structElements = new Map();
  #rawDims;
  #elementsToAddToTextLayer = null;
  #elementsToHideInTextLayer = null;
  #elementsToStealFromTextLayer = null;
  constructor(pdfPage, rawDims) {
    this.#promise = pdfPage.getStructTree();
    this.#rawDims = rawDims;
  }
  async render() {
    if (this.#treePromise) {
      return this.#treePromise;
    }
    const {
      promise,
      resolve,
      reject
    } = Promise.withResolvers();
    this.#treePromise = promise;
    try {
      const tree = await this.#promise;
      this.#collectStructElements(tree);
      this.#collectAnnotations(tree);
      this.#treeDom = this.#walk(tree);
    } catch (ex) {
      reject(ex);
    }
    this.#promise = null;
    this.#treeDom?.classList.add("structTree");
    resolve(this.#treeDom);
    return promise;
  }
  async getAriaAttributes(annotationId, {
    enableLinkOwnership = false
  } = {}) {
    try {
      await this.render();
      const ownership = this.#pendingLinkOwnership.get(annotationId);
      if (ownership && enableLinkOwnership) {
        const {
          element,
          ids
        } = ownership;
        element.removeAttribute("role");
        if (ids.length > 0) {
          this.#elementAttributes.getOrInsertComputed(annotationId, makeMap).set("aria-owns", ids.join(" "));
        }
        this.#pendingLinkOwnership.delete(annotationId);
      }
      return this.#elementAttributes.get(annotationId);
    } catch {}
    return null;
  }
  async getAnnotationIds() {
    try {
      await this.render();
      return this.#annotationIds;
    } catch {}
    return null;
  }
  hide() {
    if (this.#treeDom && !this.#treeDom.hidden) {
      this.#treeDom.hidden = true;
    }
  }
  show() {
    if (this.#treeDom?.hidden) {
      this.#treeDom.hidden = false;
    }
  }
  #collectStructElements(node) {
    if (!node) {
      return;
    }
    if (node.structId) {
      this.#structElements.getOrInsert(node.structId, node);
    }
    for (const child of node.children || []) {
      this.#collectStructElements(child);
    }
  }
  #collectAnnotations(node) {
    if (!node) {
      return;
    }
    if (node.type === "annotation") {
      this.#annotationIds.add(node.id);
      return;
    }
    for (const child of node.children || []) {
      this.#collectAnnotations(child);
    }
  }
  #getStructElementId(structId) {
    return this.#structElementIds.getOrInsertComputed(structId, () => `${this.#structElementIdPrefix}${this.#structElementIds.size}`);
  }
  #getHeaderIds(headers) {
    const result = [],
      visited = new Set(),
      pending = headers.toReversed();
    while (pending.length > 0) {
      const structId = pending.pop();
      if (visited.has(structId)) {
        continue;
      }
      visited.add(structId);
      const header = this.#structElements.get(structId);
      if (header?.role !== "TH") {
        continue;
      }
      result.push(this.#getStructElementId(structId));
      if (header.headers) {
        for (let i = header.headers.length - 1; i >= 0; i--) {
          pending.push(header.headers[i]);
        }
      }
    }
    return result;
  }
  #setAttributes(structElement, htmlElement) {
    const {
      alt,
      colSpan,
      headers,
      id,
      lang,
      rowSpan,
      short,
      structId,
      summary
    } = structElement;
    if (alt !== undefined) {
      let added = false;
      const label = removeNullCharacters(alt);
      for (const child of structElement.children) {
        if (child.type === "annotation") {
          this.#elementAttributes.getOrInsertComputed(child.id, makeMap).set("aria-label", label);
          added = true;
        }
      }
      const role = htmlElement.getAttribute("role") || (htmlElement.localName === "span" ? "generic" : null);
      if (!added && role !== "none") {
        htmlElement.setAttribute(ARIA_ROLES_WITH_PROHIBITED_NAMES.has(role) ? "aria-description" : "aria-label", label);
      }
    }
    if (id !== undefined) {
      htmlElement.setAttribute("aria-owns", id);
    }
    if (structId !== undefined && this.#structElements.get(structId) === structElement) {
      const elementId = this.#getStructElementId(structId);
      if (short !== undefined) {
        const abbreviation = document.createElement("span");
        abbreviation.setAttribute("id", elementId);
        abbreviation.setAttribute("aria-hidden", "true");
        abbreviation.textContent = removeNullCharacters(short);
        htmlElement.append(abbreviation);
      } else {
        htmlElement.setAttribute("id", elementId);
      }
    }
    if (lang !== undefined) {
      htmlElement.setAttribute("lang", removeNullCharacters(lang, true));
    }
    if (rowSpan !== undefined) {
      htmlElement.setAttribute("aria-rowspan", rowSpan);
    }
    if (colSpan !== undefined) {
      htmlElement.setAttribute("aria-colspan", colSpan);
    }
    if (headers?.length > 0) {
      const headerIds = this.#getHeaderIds(headers);
      if (headerIds.length > 0) {
        htmlElement.setAttribute("aria-describedby", headerIds.join(" "));
      }
    }
    if (summary !== undefined) {
      htmlElement.setAttribute("aria-description", removeNullCharacters(summary));
    }
  }
  #addImageInTextLayer(node, element) {
    const {
      alt,
      bbox,
      children
    } = node;
    const child = children?.[0];
    if (!this.#rawDims || !alt || !bbox || child?.type !== "content") {
      return false;
    }
    const {
      id
    } = child;
    if (!id) {
      return false;
    }
    element.setAttribute("aria-owns", id);
    const img = document.createElement("span");
    (this.#elementsToAddToTextLayer ||= new Map()).set(id, img);
    img.setAttribute("role", "img");
    img.setAttribute("aria-label", removeNullCharacters(alt));
    const {
      pageHeight,
      pageX,
      pageY
    } = this.#rawDims;
    const calc = "calc(var(--total-scale-factor) *";
    const {
      style
    } = img;
    style.width = `${calc}${bbox[2] - bbox[0]}px)`;
    style.height = `${calc}${bbox[3] - bbox[1]}px)`;
    style.left = `${calc}${bbox[0] - pageX}px)`;
    style.top = `${calc}${pageHeight - bbox[3] + pageY}px)`;
    return true;
  }
  updateTextLayer() {
    if (this.#elementsToAddToTextLayer) {
      for (const [id, img] of this.#elementsToAddToTextLayer) {
        document.getElementById(id)?.append(img);
      }
      this.#elementsToAddToTextLayer.clear();
      this.#elementsToAddToTextLayer = null;
    }
    if (this.#elementsToHideInTextLayer) {
      for (const id of this.#elementsToHideInTextLayer) {
        const elem = document.getElementById(id);
        if (elem) {
          elem.ariaHidden = true;
        }
      }
      this.#elementsToHideInTextLayer.length = 0;
      this.#elementsToHideInTextLayer = null;
    }
    if (this.#elementsToStealFromTextLayer) {
      for (let i = 0, ii = this.#elementsToStealFromTextLayer.length; i < ii; i += 2) {
        const element = this.#elementsToStealFromTextLayer[i];
        const ids = this.#elementsToStealFromTextLayer[i + 1];
        let textContent = "";
        for (const id of ids) {
          const elem = document.getElementById(id);
          if (elem) {
            textContent += elem.textContent.trim() || "";
            elem.ariaHidden = "true";
          }
        }
        if (textContent) {
          element.textContent = textContent;
        }
      }
      this.#elementsToStealFromTextLayer.length = 0;
      this.#elementsToStealFromTextLayer = null;
    }
  }
  #collectIds(node, ids) {
    if (!node) {
      return;
    }
    if ("id" in node) {
      ids.push(node.id);
    }
    for (const kid of node.children || []) {
      this.#collectIds(kid, ids);
    }
  }
  #walk(node, parentNodes = []) {
    if (!node) {
      return null;
    }
    let element;
    let visitChildren = true;
    if ("role" in node) {
      const {
        role
      } = node;
      if (MathMLElements.has(role)) {
        element = document.createElementNS(MathMLNamespace, role);
        const ids = [];
        (this.#elementsToStealFromTextLayer ||= []).push(element, ids);
        for (const {
          type,
          id
        } of node.children || []) {
          if (type === "content" && id) {
            ids.push(id);
          }
        }
      } else {
        element = document.createElement("span");
      }
      const match = role.match(HEADING_PATTERN);
      if (match) {
        element.setAttribute("role", "heading");
        element.setAttribute("aria-level", match[1]);
      } else if (PDF_ROLE_TO_HTML_ROLE[role]) {
        let htmlRole = PDF_ROLE_TO_HTML_ROLE[role];
        if (role === "TH") {
          if (node.scope === "Row") {
            htmlRole = "rowheader";
          } else if (node.scope === "Column") {
            htmlRole = "columnheader";
          } else if (parentNodes.at(-1)?.role === "TR" && parentNodes.at(-2)?.role === "TBody") {
            htmlRole = "rowheader";
          }
        } else if (role === "Caption") {
          const parentRole = parentNodes.at(-1)?.role;
          if (parentRole !== "Table" && parentRole !== "Figure") {
            htmlRole = null;
          }
        }
        if (htmlRole) {
          element.setAttribute("role", htmlRole);
        }
      }
      if (role === "Figure" && this.#addImageInTextLayer(node, element)) {
        return element;
      }
      if (role === "Formula") {
        if (node.mathML && MathMLSanitizer.sanitizer) {
          visitChildren = false;
          element.setHTML(node.mathML, {
            sanitizer: MathMLSanitizer.sanitizer
          });
          this.#collectIds(node, this.#elementsToHideInTextLayer ||= []);
          delete node.alt;
        }
        if (!node.mathML && node.children.length === 1 && node.children[0].role !== "math") {
          element = document.createElementNS(MathMLNamespace, "math");
          delete node.alt;
        }
      }
    }
    element ||= document.createElement("span");
    this.#setAttributes(node, element);
    if (node.children) {
      if (node.children.length === 1 && !("role" in node.children[0]) && "id" in node.children[0] && element.getAttribute("role") !== "none") {
        this.#setAttributes(node.children[0], element);
      } else if (visitChildren) {
        parentNodes.push(node);
        for (const kid of node.children) {
          element.append(this.#walk(kid, parentNodes));
        }
        parentNodes.pop();
      }
    }
    if (node.role === "Link") {
      const annotations = node.children?.filter(child => child.type === "annotation");
      if (annotations?.length === 1) {
        const annotation = annotations[0];
        const ids = [];
        for (const child of element.children) {
          if (child.getAttribute("aria-owns") === annotation.id) {
            continue;
          }
          child.id ||= `${this.#structElementIdPrefix}link_${this.#linkTextId++}`;
          ids.push(child.id);
        }
        this.#pendingLinkOwnership.set(annotation.id, {
          element,
          ids
        });
      }
    }
    return element;
  }
}

;// ./web/text_accessibility.js






class TextAccessibilityManager {
  #enabled = false;
  #textChildren = null;
  #textNodes = new Map();
  #waitingElements = new Map();
  setTextMapping(textDivs) {
    this.#textChildren = textDivs;
  }
  static #compareElementPositions(e1, e2) {
    const rect1 = e1.getBoundingClientRect();
    const rect2 = e2.getBoundingClientRect();
    if (rect1.width === 0 && rect1.height === 0) {
      return +1;
    }
    if (rect2.width === 0 && rect2.height === 0) {
      return -1;
    }
    const top1 = rect1.y;
    const bot1 = rect1.y + rect1.height;
    const mid1 = rect1.y + rect1.height / 2;
    const top2 = rect2.y;
    const bot2 = rect2.y + rect2.height;
    const mid2 = rect2.y + rect2.height / 2;
    if (mid1 <= top2 && mid2 >= bot1) {
      return -1;
    }
    if (mid2 <= top1 && mid1 >= bot2) {
      return +1;
    }
    const centerX1 = rect1.x + rect1.width / 2;
    const centerX2 = rect2.x + rect2.width / 2;
    return centerX1 - centerX2;
  }
  enable() {
    if (this.#enabled) {
      throw new Error("TextAccessibilityManager is already enabled.");
    }
    if (!this.#textChildren) {
      throw new Error("Text divs and strings have not been set.");
    }
    this.#enabled = true;
    this.#textChildren = this.#textChildren.slice();
    this.#textChildren.sort(TextAccessibilityManager.#compareElementPositions);
    if (this.#textNodes.size > 0) {
      const textChildren = this.#textChildren;
      for (const [id, nodeIndex] of this.#textNodes) {
        const element = document.getElementById(id);
        if (!element) {
          this.#textNodes.delete(id);
          continue;
        }
        this.#addIdToAriaOwns(id, textChildren[nodeIndex]);
      }
    }
    for (const [element, isRemovable] of this.#waitingElements) {
      this.addPointerInTextLayer(element, isRemovable);
    }
    this.#waitingElements.clear();
  }
  disable() {
    if (!this.#enabled) {
      return;
    }
    this.#waitingElements.clear();
    this.#textChildren = null;
    this.#enabled = false;
  }
  removePointerInTextLayer(element) {
    if (!this.#enabled) {
      this.#waitingElements.delete(element);
      return;
    }
    const children = this.#textChildren;
    if (!children?.length) {
      return;
    }
    const {
      id
    } = element;
    const nodeIndex = this.#textNodes.get(id);
    if (nodeIndex === undefined) {
      return;
    }
    const node = children[nodeIndex];
    this.#textNodes.delete(id);
    let owns = node.getAttribute("aria-owns");
    if (owns?.includes(id)) {
      owns = owns.split(" ").filter(x => x !== id).join(" ");
      if (owns) {
        node.setAttribute("aria-owns", owns);
      } else {
        node.removeAttribute("aria-owns");
        node.setAttribute("role", "presentation");
      }
    }
  }
  #addIdToAriaOwns(id, node) {
    const owns = node.getAttribute("aria-owns");
    if (!owns?.includes(id)) {
      node.setAttribute("aria-owns", owns ? `${owns} ${id}` : id);
    }
    node.removeAttribute("role");
  }
  addPointerInTextLayer(element, isRemovable) {
    const {
      id
    } = element;
    if (!id) {
      return null;
    }
    if (!this.#enabled) {
      this.#waitingElements.set(element, isRemovable);
      return null;
    }
    if (isRemovable) {
      this.removePointerInTextLayer(element);
    }
    const children = this.#textChildren;
    if (!children?.length) {
      return null;
    }
    const index = binarySearchFirstItem(children, node => TextAccessibilityManager.#compareElementPositions(element, node) < 0);
    const nodeIndex = Math.max(0, index - 1);
    const child = children[nodeIndex];
    this.#addIdToAriaOwns(id, child);
    this.#textNodes.set(id, nodeIndex);
    const parent = child.parentNode;
    return parent?.classList.contains("markedContent") ? parent.id : null;
  }
  moveElementInDOM(container, element, contentElement, isRemovable) {
    const id = this.addPointerInTextLayer(contentElement, isRemovable);
    if (!container.hasChildNodes()) {
      container.append(element);
      return id;
    }
    const children = Array.from(container.childNodes).filter(node => node !== element);
    if (children.length === 0) {
      return id;
    }
    const index = binarySearchFirstItem(children, node => TextAccessibilityManager.#compareElementPositions(element, node) < 0);
    if (index === 0) {
      children[0].before(element);
    } else {
      children[index - 1].after(element);
    }
    return id;
  }
}

;// ./web/text_highlighter.js



class TextHighlighter {
  #eventAC = null;
  constructor({
    findController,
    eventBus,
    pageIndex
  }) {
    this.findController = findController;
    this.matches = [];
    this.eventBus = eventBus;
    this.pageIdx = pageIndex;
    this.textDivs = null;
    this.textContentItemsStr = null;
    this.enabled = false;
  }
  setTextMapping(divs, texts) {
    this.textDivs = divs;
    this.textContentItemsStr = texts;
  }
  enable() {
    if (!this.textDivs || !this.textContentItemsStr) {
      throw new Error("Text divs and strings have not been set.");
    }
    if (this.enabled) {
      throw new Error("TextHighlighter is already enabled.");
    }
    this.enabled = true;
    if (!this.#eventAC) {
      this.#eventAC = new AbortController();
      this.eventBus.on("updatetextlayermatches", evt => {
        if (evt.pageIndex === this.pageIdx || evt.pageIndex === -1) {
          this._updateMatches();
        }
      }, {
        signal: this.#eventAC.signal,
        ...internalOpt
      });
    }
    this._updateMatches();
  }
  disable() {
    if (!this.enabled) {
      return;
    }
    this.enabled = false;
    this.#eventAC?.abort();
    this.#eventAC = null;
    this._updateMatches(true);
  }
  _convertMatches(matches, matchesLength) {
    if (!matches) {
      return [];
    }
    const {
      textContentItemsStr
    } = this;
    let i = 0,
      iIndex = 0;
    const end = textContentItemsStr.length - 1;
    const result = [];
    for (let m = 0, mm = matches.length; m < mm; m++) {
      let matchIdx = matches[m];
      while (i !== end && matchIdx >= iIndex + textContentItemsStr[i].length) {
        iIndex += textContentItemsStr[i].length;
        i++;
      }
      if (i === textContentItemsStr.length) {
        console.error("Could not find a matching mapping");
      }
      const match = {
        begin: {
          divIdx: i,
          offset: matchIdx - iIndex
        }
      };
      matchIdx += matchesLength[m];
      while (i !== end && matchIdx > iIndex + textContentItemsStr[i].length) {
        iIndex += textContentItemsStr[i].length;
        i++;
      }
      match.end = {
        divIdx: i,
        offset: matchIdx - iIndex
      };
      result.push(match);
    }
    return result;
  }
  _renderMatches(matches) {
    if (matches.length === 0) {
      return;
    }
    const {
      findController,
      pageIdx
    } = this;
    const {
      textContentItemsStr,
      textDivs
    } = this;
    const isSelectedPage = pageIdx === findController.selected.pageIdx;
    const selectedMatchIdx = findController.selected.matchIdx;
    const highlightAll = findController.state.highlightAll;
    let prevEnd = null;
    const infinity = {
      divIdx: -1,
      offset: undefined
    };
    function beginText(begin, className) {
      const divIdx = begin.divIdx;
      textDivs[divIdx].textContent = "";
      return appendTextToDiv(divIdx, 0, begin.offset, className);
    }
    function appendTextToDiv(divIdx, fromOffset, toOffset, className) {
      let div = textDivs[divIdx];
      if (div.nodeType === Node.TEXT_NODE) {
        const span = document.createElement("span");
        div.before(span);
        span.append(div);
        textDivs[divIdx] = span;
        div = span;
      }
      const content = textContentItemsStr[divIdx].substring(fromOffset, toOffset);
      const node = document.createTextNode(content);
      if (className) {
        const span = document.createElement("span");
        span.className = `${className} appended`;
        span.append(node);
        div.append(span);
        if (className.includes("selected")) {
          return span;
        }
        return null;
      }
      div.append(node);
      return 0;
    }
    let i0 = selectedMatchIdx,
      i1 = i0 + 1;
    if (highlightAll) {
      i0 = 0;
      i1 = matches.length;
    } else if (!isSelectedPage) {
      return;
    }
    let lastDivIdx = -1;
    let lastOffset = -1;
    for (let i = i0; i < i1; i++) {
      const match = matches[i];
      const begin = match.begin;
      if (begin.divIdx === lastDivIdx && begin.offset === lastOffset) {
        continue;
      }
      lastDivIdx = begin.divIdx;
      lastOffset = begin.offset;
      const end = match.end;
      const isSelected = isSelectedPage && i === selectedMatchIdx;
      const highlightSuffix = isSelected ? " selected" : "";
      let selectedSpan = null;
      if (!prevEnd || begin.divIdx !== prevEnd.divIdx) {
        if (prevEnd !== null) {
          appendTextToDiv(prevEnd.divIdx, prevEnd.offset, infinity.offset);
        }
        beginText(begin);
      } else {
        appendTextToDiv(prevEnd.divIdx, prevEnd.offset, begin.offset);
      }
      if (begin.divIdx === end.divIdx) {
        selectedSpan = appendTextToDiv(begin.divIdx, begin.offset, end.offset, "highlight" + highlightSuffix);
      } else {
        selectedSpan = appendTextToDiv(begin.divIdx, begin.offset, infinity.offset, "highlight begin" + highlightSuffix);
        for (let n0 = begin.divIdx + 1, n1 = end.divIdx; n0 < n1; n0++) {
          textDivs[n0].className = "highlight middle" + highlightSuffix;
        }
        beginText(end, "highlight end" + highlightSuffix);
      }
      prevEnd = end;
      if (isSelected) {
        findController.scrollMatchIntoView({
          element: selectedSpan,
          pageIndex: pageIdx,
          matchIndex: selectedMatchIdx
        });
      }
    }
    if (prevEnd) {
      appendTextToDiv(prevEnd.divIdx, prevEnd.offset, infinity.offset);
    }
  }
  _updateMatches(reset = false) {
    if (!this.enabled && !reset) {
      return;
    }
    const {
      findController,
      matches,
      pageIdx
    } = this;
    const {
      textContentItemsStr,
      textDivs
    } = this;
    let clearedUntilDivIdx = -1;
    for (const match of matches) {
      const begin = Math.max(clearedUntilDivIdx, match.begin.divIdx);
      for (let n = begin, end = match.end.divIdx; n <= end; n++) {
        const div = textDivs[n];
        div.textContent = textContentItemsStr[n];
        div.className = "";
      }
      clearedUntilDivIdx = match.end.divIdx + 1;
    }
    if (!findController?.highlightMatches || reset) {
      return;
    }
    const pageMatches = findController.pageMatches[pageIdx] || null;
    const pageMatchesLength = findController.pageMatchesLength[pageIdx] || null;
    this.matches = this._convertMatches(pageMatches, pageMatchesLength);
    this._renderMatches(this.matches);
  }
}

;// ./web/text_layer_builder.js














class TextLayerBuilder {
  #abortSignal = null;
  #enablePermissions = false;
  #onAppend = null;
  #renderingDone = false;
  #textLayer = null;
  static #textLayers = new Map();
  static #selectionChangeAC = null;
  constructor({
    pdfPage,
    highlighter = null,
    accessibilityManager = null,
    enablePermissions = false,
    onAppend = null,
    abortSignal = null
  }) {
    this.pdfPage = pdfPage;
    this.highlighter = highlighter;
    this.accessibilityManager = accessibilityManager;
    this.#enablePermissions = enablePermissions === true;
    this.#onAppend = onAppend;
    this.#abortSignal = abortSignal;
    this.div = document.createElement("div");
    this.div.tabIndex = 0;
    this.div.className = "textLayer";
  }
  async render({
    viewport,
    images,
    textContentParams = null
  }) {
    if (this.#renderingDone && this.#textLayer) {
      this.#textLayer.update({
        viewport,
        onBefore: this.hide.bind(this)
      });
      this.show();
      return;
    }
    this.cancel();
    this.#textLayer = new TextLayer({
      textContentSource: this.pdfPage.streamTextContent(textContentParams || {
        includeMarkedContent: true,
        disableNormalization: true
      }),
      images,
      container: this.div,
      viewport
    });
    const {
      textDivs,
      textContentItemsStr
    } = this.#textLayer;
    this.highlighter?.setTextMapping(textDivs, textContentItemsStr);
    this.accessibilityManager?.setTextMapping(textDivs);
    await this.#textLayer.render();
    this.#renderingDone = true;
    const endOfContent = document.createElement("div");
    endOfContent.className = "endOfContent";
    this.div.append(endOfContent);
    this.#bindMouse(endOfContent);
    this.#onAppend?.(this.div);
    this.highlighter?.enable();
    this.accessibilityManager?.enable();
  }
  hide() {
    if (!this.div.hidden && this.#renderingDone) {
      this.highlighter?.disable();
      this.div.hidden = true;
    }
  }
  show() {
    if (this.div.hidden && this.#renderingDone) {
      this.div.hidden = false;
      this.highlighter?.enable();
    }
  }
  cancel() {
    this.#textLayer?.cancel();
    this.#textLayer = null;
    this.#renderingDone = false;
    this.div.replaceChildren();
    this.highlighter?.disable();
    this.accessibilityManager?.disable();
    TextLayerBuilder.#removeGlobalSelectionListener(this.div);
  }
  #bindMouse(end) {
    const {
      div
    } = this;
    const abortSignal = this.#abortSignal;
    const opts = abortSignal ? {
      signal: abortSignal
    } : null;
    div.addEventListener("mousedown", () => {
      div.classList.add("selecting");
    }, opts);
    div.addEventListener("copy", event => {
      if (!this.#enablePermissions) {
        const selection = document.getSelection();
        event.clipboardData.setData("text/plain", removeNullCharacters(normalizeUnicode(selection.toString())));
      }
      stopEvent(event);
    }, opts);
    TextLayerBuilder.#textLayers.set(div, end);
    TextLayerBuilder.#enableGlobalSelectionListener(abortSignal);
  }
  static #removeGlobalSelectionListener(textLayerDiv) {
    this.#textLayers.delete(textLayerDiv);
    if (this.#textLayers.size === 0) {
      this.#selectionChangeAC?.abort();
      this.#selectionChangeAC = null;
    }
  }
  static #enableGlobalSelectionListener(globalAbortSignal) {
    if (this.#selectionChangeAC) {
      return;
    }
    this.#selectionChangeAC = new AbortController();
    const signal = globalAbortSignal ? AbortSignal.any([this.#selectionChangeAC.signal, globalAbortSignal]) : this.#selectionChangeAC.signal;
    const reset = (end, textLayer) => {
      textLayer.append(end);
      end.style.width = "";
      end.style.height = "";
      textLayer.classList.remove("selecting");
    };
    let isPointerDown = false;
    document.addEventListener("pointerdown", () => {
      isPointerDown = true;
    }, {
      signal
    });
    document.addEventListener("pointerup", () => {
      isPointerDown = false;
      this.#textLayers.forEach(reset);
    }, {
      signal
    });
    window.addEventListener("blur", () => {
      isPointerDown = false;
      this.#textLayers.forEach(reset);
    }, {
      signal
    });
    document.addEventListener("keyup", () => {
      if (!isPointerDown) {
        this.#textLayers.forEach(reset);
      }
    }, {
      signal
    });
    var isFirefoxOrModernChromium, prevRange;
    document.addEventListener("selectionchange", () => {
      const selection = document.getSelection();
      if (selection.rangeCount === 0) {
        this.#textLayers.forEach(reset);
        return;
      }
      const activeTextLayers = new Set();
      for (let i = 0; i < selection.rangeCount; i++) {
        const range = selection.getRangeAt(i);
        for (const textLayerDiv of this.#textLayers.keys()) {
          if (!activeTextLayers.has(textLayerDiv) && range.intersectsNode(textLayerDiv)) {
            activeTextLayers.add(textLayerDiv);
          }
        }
      }
      for (const [textLayerDiv, endDiv] of this.#textLayers) {
        if (activeTextLayers.has(textLayerDiv)) {
          textLayerDiv.classList.add("selecting");
        } else {
          reset(endDiv, textLayerDiv);
        }
      }
      if (isFirefoxOrModernChromium === undefined) {
        isFirefoxOrModernChromium = getComputedStyle(this.#textLayers.values().next().value).getPropertyValue("-moz-user-select") === "none";
        if (!isFirefoxOrModernChromium) {
          const chromiumVersion = navigator.userAgentData ? navigator.userAgentData.brands.find(({
            brand
          }) => brand === "Chromium")?.version : /\bChrome\/(\d+)\b/.exec(navigator.userAgent)?.[1];
          isFirefoxOrModernChromium = !!chromiumVersion && parseInt(chromiumVersion, 10) >= 148;
        }
      }
      if (isFirefoxOrModernChromium) {
        return;
      }
      const range = selection.getRangeAt(0);
      const modifyStart = prevRange && (range.compareBoundaryPoints(Range.END_TO_END, prevRange) === 0 || range.compareBoundaryPoints(Range.START_TO_END, prevRange) === 0);
      let anchor = modifyStart ? range.startContainer : range.endContainer;
      if (anchor.nodeType === Node.TEXT_NODE) {
        anchor = anchor.parentNode;
      }
      if (anchor.classList?.contains("highlight")) {
        anchor = anchor.parentNode;
      }
      if (!modifyStart && range.endOffset === 0) {
        do {
          while (!anchor.previousSibling) {
            anchor = anchor.parentNode;
          }
          anchor = anchor.previousSibling;
        } while (!anchor.childNodes.length);
      }
      const parentTextLayer = anchor.parentElement?.closest(".textLayer");
      const endDiv = this.#textLayers.get(parentTextLayer);
      if (endDiv) {
        endDiv.style.width = parentTextLayer.style.width;
        endDiv.style.height = parentTextLayer.style.height;
        endDiv.style.userSelect = "text";
        anchor.parentElement.insertBefore(endDiv, modifyStart ? anchor : anchor.nextSibling);
      }
      prevRange = range.cloneRange();
    }, {
      signal
    });
  }
}

;// ./web/xfa_layer_builder.js

class XfaLayerBuilder {
  #cancelled = false;
  div = null;
  constructor({
    pdfPage,
    annotationStorage = null,
    linkService,
    xfaHtml = null
  }) {
    this.pdfPage = pdfPage;
    this.annotationStorage = annotationStorage;
    this.linkService = linkService;
    this.xfaHtml = xfaHtml;
  }
  async render({
    viewport,
    intent = "display"
  }) {
    let xfaHtml;
    if (intent === "print") {
      xfaHtml = this.xfaHtml;
    } else {
      xfaHtml = await this.pdfPage.getXfa();
      if (this.#cancelled || !xfaHtml) {
        return {
          textDivs: []
        };
      }
    }
    const hasDiv = !!this.div;
    const params = {
      viewport: viewport.clone({
        dontFlip: true
      }),
      div: this.div ??= document.createElement("div"),
      xfaHtml,
      annotationStorage: this.annotationStorage,
      linkService: this.linkService,
      intent
    };
    return hasDiv ? XfaLayer.update(params) : XfaLayer.render(params);
  }
  cancel() {
    this.#cancelled = true;
  }
  hide() {
    if (!this.div) {
      return;
    }
    this.div.hidden = true;
  }
}

;// ./web/pdf_page_view.js





















const DEFAULT_LAYER_PROPERTIES = {
  annotationEditorUIManager: null,
  annotationStorage: null,
  downloadManager: null,
  enableScripting: false,
  fieldObjectsPromise: null,
  findController: null,
  hasJSActionsPromise: null,
  get linkService() {
    return new SimpleLinkService();
  }
};
const LAYERS_ORDER = new Map([["canvasWrapper", 0], ["textLayer", 1], ["annotationLayer", 2], ["annotationEditorLayer", 3], ["xfaLayer", 3]]);
class PDFPageView extends BasePDFPageView {
  #abortSignal = null;
  #annotationMode = AnnotationMode.ENABLE_FORMS;
  #canvasWrapper = null;
  #commentManager = null;
  #enableAutoLinking = true;
  #hasRestrictedScaling = false;
  #isEditing = false;
  #layerProperties = null;
  #needsRestrictedScaling = false;
  #originalViewport = null;
  #previousRotation = null;
  #scaleRoundX = 1;
  #scaleRoundY = 1;
  #textLayerMode = TextLayerMode.ENABLE;
  #userUnit = 1;
  #useThumbnailCanvas = {
    directDrawing: true,
    initialOptionalContent: true,
    regularAnnotations: true
  };
  #layers = [null, null, null, null];
  constructor(options) {
    super(options);
    const {
      container,
      defaultViewport
    } = options;
    this.renderingId = "page" + this.id;
    this.#layerProperties = options.layerProperties || DEFAULT_LAYER_PROPERTIES;
    this.#abortSignal = options.abortSignal || null;
    this.pdfPage = null;
    this.pageLabel = null;
    this.rotation = 0;
    this.scale = options.scale || (/* inlined export .DEFAULT_SCALE */1);
    this.viewport = defaultViewport;
    this.pdfPageRotate = defaultViewport.rotation;
    this._optionalContentConfigPromise = options.optionalContentConfigPromise || null;
    this.#textLayerMode = options.textLayerMode ?? TextLayerMode.ENABLE;
    this.#annotationMode = options.annotationMode ?? AnnotationMode.ENABLE_FORMS;
    this.imageResourcesPath = options.imageResourcesPath || "";
    this.enableDetailCanvas = options.enableDetailCanvas ?? true;
    this.maxCanvasPixels = options.maxCanvasPixels ?? AppOptions.get("maxCanvasPixels");
    this.maxCanvasDim = options.maxCanvasDim || AppOptions.get("maxCanvasDim");
    this.capCanvasAreaFactor = options.capCanvasAreaFactor ?? AppOptions.get("capCanvasAreaFactor");
    this.#enableAutoLinking = options.enableAutoLinking !== false;
    this.#commentManager = options.commentManager || null;
    this.l10n = options.l10n;
    this.l10n ||= new genericl10n_GenericL10n();
    this._isStandalone = !this.renderingQueue?.hasViewer();
    this._container = container;
    this._annotationCanvasMap = null;
    this.annotationLayer = null;
    this.annotationEditorLayer = null;
    this.textLayer = null;
    this.xfaLayer = null;
    this.structTreeLayer = null;
    this.drawLayer = null;
    this.detailView = null;
    const div = document.createElement("div");
    div.className = "page";
    div.setAttribute("data-page-number", this.id);
    div.setAttribute("role", "region");
    div.setAttribute("data-l10n-id", "pdfjs-page-landmark");
    div.setAttribute("data-l10n-args", JSON.stringify({
      page: this.id
    }));
    this.div = div;
    this.#setDimensions();
    container?.append(div);
    if (this._isStandalone) {
      container?.style.setProperty("--scale-factor", this.scale * PixelsPerInch.PDF_TO_CSS_UNITS);
      if (this.pageColors?.background) {
        container?.style.setProperty("--page-bg-color", this.pageColors.background);
      }
      const {
        optionalContentConfigPromise
      } = options;
      if (optionalContentConfigPromise) {
        optionalContentConfigPromise.then(optionalContentConfig => {
          if (optionalContentConfigPromise !== this._optionalContentConfigPromise) {
            return;
          }
          this.#useThumbnailCanvas.initialOptionalContent = optionalContentConfig.hasInitialVisibility;
        });
      }
      if (!options.l10n) {
        this.l10n.translate(this.div);
      }
    }
  }
  clone(id) {
    const clone = new PDFPageView({
      container: null,
      eventBus: this.eventBus,
      pagesColors: this.pageColors,
      renderingQueue: this.renderingQueue,
      enableOptimizedPartialRendering: this.enableOptimizedPartialRendering,
      minDurationToUpdateCanvas: this.minDurationToUpdateCanvas,
      defaultViewport: this.viewport,
      id,
      layerProperties: this.#layerProperties,
      abortSignal: this.#abortSignal,
      scale: this.scale,
      optionalContentConfigPromise: this._optionalContentConfigPromise,
      textLayerMode: this.#textLayerMode,
      annotationMode: this.#annotationMode,
      imageResourcesPath: this.imageResourcesPath,
      enableDetailCanvas: this.enableDetailCanvas,
      maxCanvasPixels: this.maxCanvasPixels,
      maxCanvasDim: this.maxCanvasDim,
      capCanvasAreaFactor: this.capCanvasAreaFactor,
      enableAutoLinking: this.#enableAutoLinking,
      commentManager: this.#commentManager,
      l10n: this.l10n
    });
    clone.setPdfPage(this.pdfPage.clone(id - 1));
    return clone;
  }
  #addLayer(div, name) {
    const pos = LAYERS_ORDER.get(name);
    const oldDiv = this.#layers[pos];
    this.#layers[pos] = div;
    if (oldDiv) {
      oldDiv.replaceWith(div);
      return;
    }
    for (let i = pos - 1; i >= 0; i--) {
      const layer = this.#layers[i];
      if (layer) {
        layer.after(div);
        return;
      }
    }
    this.div.prepend(div);
  }
  #setDimensions() {
    const {
      div,
      viewport
    } = this;
    if (viewport.userUnit !== this.#userUnit) {
      if (viewport.userUnit !== 1) {
        div.style.setProperty("--user-unit", viewport.userUnit);
      } else {
        div.style.removeProperty("--user-unit");
      }
      this.#userUnit = viewport.userUnit;
    }
    if (this.pdfPage) {
      if (this.#previousRotation === viewport.rotation) {
        return;
      }
      this.#previousRotation = viewport.rotation;
    }
    setLayerDimensions(div, viewport, true, false);
  }
  updatePageNumber(newPageNumber) {
    const oldPageNumber = this.id;
    if (oldPageNumber !== newPageNumber) {
      this.id = newPageNumber;
      this.renderingId = `page${newPageNumber}`;
      if (this.pdfPage) {
        this.pdfPage.pageNumber = newPageNumber;
      }
      this.setPageLabel(this.pageLabel);
      const {
        div
      } = this;
      div.setAttribute("data-page-number", newPageNumber);
      div.setAttribute("data-l10n-args", JSON.stringify({
        page: newPageNumber
      }));
      this._textHighlighter.pageIdx = newPageNumber - 1;
    }
    this.#layerProperties.annotationEditorUIManager?.updatePageIndex(oldPageNumber - 1, newPageNumber - 1);
  }
  setPdfPage(pdfPage) {
    if (this._isStandalone && (this.pageColors?.foreground === "CanvasText" || this.pageColors?.background === "Canvas")) {
      this._container?.style.setProperty("--hcm-highlight-filter", pdfPage.filterFactory.addHighlightHCMFilter("highlight", "CanvasText", "Canvas", "HighlightText", "Highlight"));
      this._container?.style.setProperty("--hcm-highlight-selected-filter", pdfPage.filterFactory.addHighlightHCMFilter("highlight_selected", "CanvasText", "Canvas", "HighlightText", "Highlight"));
    }
    this.pdfPage = pdfPage;
    this.pdfPageRotate = pdfPage.rotate;
    const totalRotation = (this.rotation + this.pdfPageRotate) % 360;
    this.viewport = pdfPage.getViewport({
      scale: this.scale * PixelsPerInch.PDF_TO_CSS_UNITS,
      rotation: totalRotation
    });
    this.#setDimensions();
    this.reset();
  }
  destroy() {
    this.reset();
    this.pdfPage?.cleanup();
  }
  deleteMe(isCut) {
    if (isCut) {
      this.div.remove();
      return;
    }
    this.destroy();
    this.#layerProperties.annotationEditorUIManager?.deletePage(this.id);
  }
  hasEditableAnnotations() {
    return !!this.annotationLayer?.hasEditableAnnotations();
  }
  get _textHighlighter() {
    return shadow(this, "_textHighlighter", new TextHighlighter({
      pageIndex: this.id - 1,
      eventBus: this.eventBus,
      findController: this.#layerProperties.findController
    }));
  }
  #dispatchLayerRendered(name, error) {
    this.eventBus.dispatch(name, {
      source: this,
      pageNumber: this.id,
      error
    });
  }
  async #renderAnnotationLayer(textLayerPromise = null) {
    const {
      annotationLayer,
      textLayer
    } = this;
    let error = null;
    try {
      await this.annotationLayer.render({
        viewport: this.viewport,
        intent: "display",
        structTreeLayer: this.structTreeLayer,
        optionalContentConfigPromise: this._optionalContentConfigPromise
      });
    } catch (ex) {
      console.error("#renderAnnotationLayer:", ex);
      error = ex;
    } finally {
      this.#dispatchLayerRendered("annotationlayerrendered", error);
    }
    if (this.#enableAutoLinking && textLayerPromise) {
      this.#injectLinkAnnotations(textLayerPromise, annotationLayer, textLayer);
    }
  }
  _refreshAnnotationLayer() {
    if (this._annotationCanvasMap?.size) {
      this.annotationLayer?.refreshCanvases();
    }
  }
  async #renderAnnotationEditorLayer() {
    let error = null;
    try {
      await this.annotationEditorLayer.render({
        viewport: this.viewport,
        intent: "display"
      });
    } catch (ex) {
      console.error("#renderAnnotationEditorLayer:", ex);
      error = ex;
    } finally {
      this.#dispatchLayerRendered("annotationeditorlayerrendered", error);
    }
  }
  async #renderDrawLayer() {
    try {
      await this.drawLayer.render({
        intent: "display"
      });
    } catch (ex) {
      console.error("#renderDrawLayer:", ex);
    }
  }
  async #renderXfaLayer() {
    let error = null;
    try {
      const result = await this.xfaLayer.render({
        viewport: this.viewport,
        intent: "display"
      });
      if (result?.textDivs && this._textHighlighter) {
        this.#buildXfaTextContentItems(result.textDivs);
      }
    } catch (ex) {
      console.error("#renderXfaLayer:", ex);
      error = ex;
    } finally {
      if (this.xfaLayer?.div) {
        this.l10n.pause();
        this.#addLayer(this.xfaLayer.div, "xfaLayer");
        this.l10n.resume();
      }
      this.#dispatchLayerRendered("xfalayerrendered", error);
    }
  }
  async #renderTextLayer() {
    let error = null;
    try {
      await this.textLayer.render({
        viewport: this.viewport,
        images: this.imageCoordinates ? new TextLayerImages(this.imagesRightClickMinSize, this.imageCoordinates, this.viewport, () => this.canvas) : null
      });
    } catch (ex) {
      if (ex instanceof AbortException) {
        return;
      }
      console.error("#renderTextLayer:", ex);
      error = ex;
    }
    this.#dispatchLayerRendered("textlayerrendered", error);
    this.#renderStructTreeLayer();
  }
  async #renderStructTreeLayer() {
    const treeDom = await this.structTreeLayer?.render();
    if (treeDom) {
      this.l10n.pause();
      this.structTreeLayer?.updateTextLayer();
      if (this.canvas && treeDom.parentNode !== this.canvas) {
        this.canvas.append(treeDom);
      }
      this.l10n.resume();
    }
    this.structTreeLayer?.show();
  }
  async #buildXfaTextContentItems(textDivs) {
    const text = await this.pdfPage.getTextContent();
    const items = [];
    for (const item of text.items) {
      items.push(item.str);
    }
    this._textHighlighter.setTextMapping(textDivs, items);
    this._textHighlighter.enable();
  }
  async #injectLinkAnnotations(textLayerPromise, annotationLayer, textLayer) {
    let error = null;
    try {
      await textLayerPromise;
      if (annotationLayer !== this.annotationLayer || textLayer !== this.textLayer) {
        return;
      }
      await annotationLayer.injectLinkAnnotations(Autolinker.processLinks(this));
    } catch (ex) {
      console.error("#injectLinkAnnotations:", ex);
      error = ex;
    }
    this.#dispatchLayerRendered("linkannotationsadded", error);
  }
  _resetCanvas() {
    super._resetCanvas();
    this.#originalViewport = null;
  }
  reset({
    keepAnnotationLayer = false,
    keepAnnotationEditorLayer = false,
    keepXfaLayer = false,
    keepTextLayer = false,
    keepCanvasWrapper = false,
    preserveDetailViewState = false
  } = {}) {
    const keepPdfBugGroups = this.pdfPage?._pdfBug ?? false;
    this.cancelRendering({
      keepAnnotationLayer,
      keepAnnotationEditorLayer,
      keepXfaLayer,
      keepTextLayer
    });
    this.renderingState = RenderingStates.INITIAL;
    const div = this.div;
    const childNodes = div.childNodes,
      annotationLayerNode = keepAnnotationLayer && this.annotationLayer?.div || null,
      annotationEditorLayerNode = keepAnnotationEditorLayer && this.annotationEditorLayer?.div || null,
      xfaLayerNode = keepXfaLayer && this.xfaLayer?.div || null,
      textLayerNode = keepTextLayer && this.textLayer?.div || null,
      canvasWrapperNode = keepCanvasWrapper && this.#canvasWrapper || null;
    for (let i = childNodes.length - 1; i >= 0; i--) {
      const node = childNodes[i];
      switch (node) {
        case annotationLayerNode:
        case annotationEditorLayerNode:
        case xfaLayerNode:
        case textLayerNode:
        case canvasWrapperNode:
          continue;
      }
      if (keepPdfBugGroups && node.classList.contains("pdfBugGroupsLayer")) {
        continue;
      }
      node.remove();
      const layerIndex = this.#layers.indexOf(node);
      if (layerIndex >= 0) {
        this.#layers[layerIndex] = null;
      }
    }
    div.removeAttribute("data-loaded");
    if (annotationLayerNode) {
      this.annotationLayer.hide();
    }
    if (annotationEditorLayerNode) {
      this.annotationEditorLayer.hide();
    }
    if (xfaLayerNode) {
      this.xfaLayer.hide();
    }
    if (textLayerNode) {
      this.textLayer.hide();
    }
    this.structTreeLayer?.hide();
    if (!keepCanvasWrapper && this.#canvasWrapper) {
      this.#canvasWrapper = null;
      this._resetCanvas();
    }
    if (!preserveDetailViewState) {
      this.detailView?.reset({
        keepCanvas: keepCanvasWrapper
      });
      if (!keepCanvasWrapper) {
        this.detailView = null;
      }
    }
  }
  toggleEditingMode(isEditing) {
    this.#isEditing = isEditing;
    if (!this.hasEditableAnnotations()) {
      return;
    }
    this.reset({
      keepAnnotationLayer: true,
      keepAnnotationEditorLayer: true,
      keepXfaLayer: true,
      keepTextLayer: true,
      keepCanvasWrapper: true
    });
  }
  updateVisibleArea(visibleArea) {
    if (this.enableDetailCanvas) {
      if (this.#needsRestrictedScaling && this.maxCanvasPixels > 0 && visibleArea) {
        this.detailView ??= new PDFPageDetailView({
          pageView: this,
          enableOptimizedPartialRendering: this.enableOptimizedPartialRendering,
          imagesRightClickMinSize: -1
        });
        this.detailView.update({
          visibleArea
        });
      } else if (this.detailView) {
        this.detailView.reset();
        this.detailView = null;
      }
    }
  }
  update({
    scale = 0,
    rotation = null,
    optionalContentConfigPromise = null,
    drawingDelay = -1
  }) {
    this.scale = scale || this.scale;
    if (typeof rotation === "number") {
      this.rotation = rotation;
    }
    if (optionalContentConfigPromise instanceof Promise) {
      this._optionalContentConfigPromise = optionalContentConfigPromise;
      optionalContentConfigPromise.then(optionalContentConfig => {
        if (optionalContentConfigPromise !== this._optionalContentConfigPromise) {
          return;
        }
        this.#useThumbnailCanvas.initialOptionalContent = optionalContentConfig.hasInitialVisibility;
      });
    }
    this.#useThumbnailCanvas.directDrawing = true;
    const totalRotation = (this.rotation + this.pdfPageRotate) % 360;
    this.viewport = this.viewport.clone({
      scale: this.scale * PixelsPerInch.PDF_TO_CSS_UNITS,
      rotation: totalRotation
    });
    this.#setDimensions();
    if (this._isStandalone) {
      this._container?.style.setProperty("--scale-factor", this.viewport.scale);
    }
    this.#computeScale();
    if (this.canvas) {
      const onlyCssZoom = this.#hasRestrictedScaling && this.#needsRestrictedScaling;
      const postponeDrawing = drawingDelay >= 0 && drawingDelay < 1000;
      if (postponeDrawing || onlyCssZoom) {
        if (postponeDrawing && !onlyCssZoom && this.renderingState !== RenderingStates.FINISHED) {
          this.cancelRendering({
            keepAnnotationLayer: true,
            keepAnnotationEditorLayer: true,
            keepXfaLayer: true,
            keepTextLayer: true,
            cancelExtraDelay: drawingDelay
          });
          this.renderingState = RenderingStates.FINISHED;
          this.#useThumbnailCanvas.directDrawing = false;
        }
        this.cssTransform({
          redrawAnnotationLayer: true,
          redrawAnnotationEditorLayer: true,
          redrawXfaLayer: true,
          redrawTextLayer: !postponeDrawing,
          hideTextLayer: postponeDrawing
        });
        if (!postponeDrawing) {
          this.detailView?.update({
            underlyingViewUpdated: true
          });
          this.dispatchPageRendered(true, false);
        }
        return;
      }
    }
    this.cssTransform({});
    this.annotationEditorLayer?.update(this.viewport);
    this.reset({
      keepAnnotationLayer: true,
      keepAnnotationEditorLayer: true,
      keepXfaLayer: true,
      keepTextLayer: true,
      keepCanvasWrapper: true,
      preserveDetailViewState: true
    });
    this.detailView?.update({
      underlyingViewUpdated: true
    });
  }
  #computeScale() {
    const {
      width,
      height
    } = this.viewport;
    const outputScale = this.outputScale = new OutputScale();
    if (this.maxCanvasPixels === 0) {
      const invScale = 1 / this.scale;
      outputScale.sx *= invScale;
      outputScale.sy *= invScale;
      this.#needsRestrictedScaling = true;
    } else {
      this.#needsRestrictedScaling = outputScale.limitCanvas(width, height, this.maxCanvasPixels, this.maxCanvasDim, this.capCanvasAreaFactor);
      if (this.#needsRestrictedScaling && this.enableDetailCanvas) {
        const factor = this.enableOptimizedPartialRendering ? 4 : 2;
        outputScale.sx /= factor;
        outputScale.sy /= factor;
      }
    }
  }
  cancelRendering({
    keepAnnotationLayer = false,
    keepAnnotationEditorLayer = false,
    keepXfaLayer = false,
    keepTextLayer = false,
    cancelExtraDelay = 0
  } = {}) {
    super.cancelRendering({
      cancelExtraDelay
    });
    if (this.textLayer && (!keepTextLayer || !this.textLayer.div)) {
      this.textLayer.cancel();
      this.textLayer = null;
    }
    if (this.annotationLayer && (!keepAnnotationLayer || !this.annotationLayer.div)) {
      this.annotationLayer.cancel();
      this.annotationLayer = null;
      this._annotationCanvasMap = null;
    }
    if (this.structTreeLayer && !this.textLayer) {
      this.structTreeLayer = null;
    }
    if (this.annotationEditorLayer && (!keepAnnotationEditorLayer || !this.annotationEditorLayer.div || !this.textLayer)) {
      if (this.drawLayer) {
        this.drawLayer.cancel();
        this.drawLayer = null;
      }
      this.annotationEditorLayer.cancel();
      this.annotationEditorLayer = null;
    }
    if (this.drawLayer && !this.textLayer) {
      this.drawLayer.cancel();
      this.drawLayer = null;
    }
    if (this.xfaLayer && (!keepXfaLayer || !this.xfaLayer.div)) {
      this.xfaLayer.cancel();
      this.xfaLayer = null;
      this._textHighlighter?.disable();
    }
  }
  cssTransform({
    redrawAnnotationLayer = false,
    redrawAnnotationEditorLayer = false,
    redrawXfaLayer = false,
    redrawTextLayer = false,
    hideTextLayer = false
  }) {
    const {
      canvas
    } = this;
    if (!canvas) {
      return;
    }
    const originalViewport = this.#originalViewport;
    if (this.viewport !== originalViewport) {
      const relativeRotation = (360 + this.viewport.rotation - originalViewport.rotation) % 360;
      if (relativeRotation === 90 || relativeRotation === 270) {
        const {
          width,
          height
        } = this.viewport;
        const scaleX = height / width;
        const scaleY = width / height;
        canvas.style.transform = `rotate(${relativeRotation}deg) scale(${scaleX},${scaleY})`;
      } else {
        canvas.style.transform = relativeRotation === 0 ? "" : `rotate(${relativeRotation}deg)`;
      }
    }
    if (redrawAnnotationLayer && this.annotationLayer) {
      this.#renderAnnotationLayer();
    }
    if (redrawAnnotationEditorLayer && this.annotationEditorLayer) {
      if (this.drawLayer) {
        this.#renderDrawLayer();
      }
      this.#renderAnnotationEditorLayer();
    }
    if (redrawXfaLayer && this.xfaLayer) {
      this.#renderXfaLayer();
    }
    if (this.textLayer) {
      if (hideTextLayer) {
        this.textLayer.hide();
        this.structTreeLayer?.hide();
      } else if (redrawTextLayer) {
        this.#renderTextLayer();
      }
    }
  }
  get width() {
    return this.viewport.width;
  }
  get height() {
    return this.viewport.height;
  }
  getPagePoint(x, y) {
    return this.viewport.convertToPdfPoint(x, y);
  }
  _ensureCanvasWrapper() {
    let canvasWrapper = this.#canvasWrapper;
    if (!canvasWrapper) {
      canvasWrapper = this.#canvasWrapper = document.createElement("div");
      canvasWrapper.classList.add("canvasWrapper");
      this.#addLayer(canvasWrapper, "canvasWrapper");
    }
    return canvasWrapper;
  }
  _getRenderingContext(canvas, transform, recordOperations, recordImages) {
    return {
      canvas,
      transform,
      viewport: this.viewport,
      annotationMode: this.#annotationMode,
      optionalContentConfigPromise: this._optionalContentConfigPromise,
      annotationCanvasMap: this._annotationCanvasMap,
      pageColors: this.pageColors,
      isEditing: this.#isEditing,
      recordOperations,
      recordImages
    };
  }
  async draw() {
    if (this.renderingState !== RenderingStates.INITIAL) {
      console.error("Must be in new state before drawing");
      this.reset();
    }
    const {
      div,
      l10n,
      pdfPage,
      viewport
    } = this;
    if (!pdfPage) {
      this.renderingState = RenderingStates.FINISHED;
      throw new Error("pdfPage is not loaded");
    }
    this.renderingState = RenderingStates.RUNNING;
    const canvasWrapper = this._ensureCanvasWrapper();
    if (!this.textLayer && this.#textLayerMode !== TextLayerMode.DISABLE && !pdfPage.isPureXfa) {
      this._accessibilityManager ||= new TextAccessibilityManager();
      this.textLayer = new TextLayerBuilder({
        pdfPage,
        highlighter: this._textHighlighter,
        accessibilityManager: this._accessibilityManager,
        enablePermissions: this.#textLayerMode === TextLayerMode.ENABLE_PERMISSIONS,
        onAppend: textLayerDiv => {
          this.l10n.pause();
          this.#addLayer(textLayerDiv, "textLayer");
          this.l10n.resume();
        },
        abortSignal: this.#abortSignal
      });
      if (this.enableSelectionRendering) {
        this.textLayer.div.classList.add("selectionRendering");
      }
    }
    if (!this.annotationLayer && this.#annotationMode !== AnnotationMode.DISABLE) {
      const {
        annotationStorage,
        annotationEditorUIManager,
        downloadManager,
        enableComment,
        enableScripting,
        fieldObjectsPromise,
        hasJSActionsPromise,
        linkService
      } = this.#layerProperties;
      this._annotationCanvasMap ||= new Map();
      this.annotationLayer = new AnnotationLayerBuilder({
        pdfPage,
        annotationStorage,
        imageResourcesPath: this.imageResourcesPath,
        renderForms: this.#annotationMode === AnnotationMode.ENABLE_FORMS,
        linkService,
        downloadManager,
        enableComment,
        enableScripting,
        hasJSActionsPromise,
        fieldObjectsPromise,
        annotationCanvasMap: this._annotationCanvasMap,
        accessibilityManager: this._accessibilityManager,
        annotationEditorUIManager,
        commentManager: this.#commentManager,
        onAppend: annotationLayerDiv => {
          this.#addLayer(annotationLayerDiv, "annotationLayer");
        }
      });
    }
    const {
      width,
      height
    } = viewport;
    this.#originalViewport = viewport;
    const {
      canvas,
      prevCanvas
    } = this._createCanvas(newCanvas => {
      canvasWrapper.prepend(newCanvas);
    });
    canvas.setAttribute("role", "presentation");
    if (!this.outputScale) {
      this.#computeScale();
    }
    const {
      outputScale
    } = this;
    this.#hasRestrictedScaling = this.#needsRestrictedScaling;
    const sfx = approximateFraction(outputScale.sx);
    const sfy = approximateFraction(outputScale.sy);
    const canvasWidth = canvas.width = floorToDivide(calcRound(width * outputScale.sx), sfx[0]);
    const canvasHeight = canvas.height = floorToDivide(calcRound(height * outputScale.sy), sfy[0]);
    const pageWidth = floorToDivide(calcRound(width), sfx[1]);
    const pageHeight = floorToDivide(calcRound(height), sfy[1]);
    outputScale.sx = canvasWidth / pageWidth;
    outputScale.sy = canvasHeight / pageHeight;
    if (this.#scaleRoundX !== sfx[1]) {
      div.style.setProperty("--scale-round-x", `${sfx[1]}px`);
      this.#scaleRoundX = sfx[1];
    }
    if (this.#scaleRoundY !== sfy[1]) {
      div.style.setProperty("--scale-round-y", `${sfy[1]}px`);
      this.#scaleRoundY = sfy[1];
    }
    const recordBBoxes = this.enableOptimizedPartialRendering && this.#hasRestrictedScaling && !this.recordedBBoxes;
    const recordImages = this.imagesRightClickMinSize !== -1 && !this.imageCoordinates;
    const transform = outputScale.scaled ? [outputScale.sx, 0, 0, outputScale.sy, 0, 0] : null;
    const resultPromise = this._drawCanvas(this._getRenderingContext(canvas, transform, recordBBoxes, recordImages), () => {
      prevCanvas?.remove();
      this._resetCanvas();
    }, renderTask => {
      this.#useThumbnailCanvas.regularAnnotations = !renderTask.separateAnnots;
      this.dispatchPageRendered(false, false);
    }).then(async () => {
      if (this.renderingState !== RenderingStates.FINISHED) {
        return;
      }
      this.structTreeLayer ||= new StructTreeLayerBuilder(pdfPage, viewport.rawDims);
      const textLayerPromise = this.textLayer ? this.#renderTextLayer() : null;
      if (this.annotationLayer) {
        await this.#renderAnnotationLayer(textLayerPromise);
      }
      this.drawLayer ||= new DrawLayerBuilder({
        pageIndex: this.id,
        textLayer: this.enableSelectionRendering ? this.textLayer?.div : null,
        filterFactory: this.pdfPage?.filterFactory,
        pageColors: this.pageColors
      });
      await this.#renderDrawLayer();
      this.drawLayer.setParent(canvasWrapper);
      const {
        annotationEditorUIManager
      } = this.#layerProperties;
      if (!annotationEditorUIManager) {
        return;
      }
      if (this.annotationLayer || this.#annotationMode === AnnotationMode.DISABLE) {
        this.annotationEditorLayer ||= new AnnotationEditorLayerBuilder({
          uiManager: annotationEditorUIManager,
          pageIndex: this.id - 1,
          l10n,
          structTreeLayer: this.structTreeLayer,
          accessibilityManager: this._accessibilityManager,
          annotationLayer: this.annotationLayer?.annotationLayer,
          textLayer: this.textLayer,
          drawLayer: this.drawLayer.getDrawLayer(),
          onAppend: annotationEditorLayerDiv => {
            this.#addLayer(annotationEditorLayerDiv, "annotationEditorLayer");
          }
        });
        this.#renderAnnotationEditorLayer();
      }
    });
    if (pdfPage.isPureXfa) {
      if (!this.xfaLayer) {
        const {
          annotationStorage,
          linkService
        } = this.#layerProperties;
        this.xfaLayer = new XfaLayerBuilder({
          pdfPage,
          annotationStorage,
          linkService
        });
      }
      this.#renderXfaLayer();
    }
    div.setAttribute("data-loaded", true);
    this.dispatchPageRender();
    return resultPromise;
  }
  setPageLabel(label) {
    this.pageLabel = typeof label === "string" ? label : null;
    this.div.setAttribute("data-l10n-args", JSON.stringify({
      page: this.pageLabel ?? this.id
    }));
    if (this.pageLabel !== null) {
      this.div.setAttribute("data-page-label", this.pageLabel);
    } else {
      this.div.removeAttribute("data-page-label");
    }
  }
  get thumbnailCanvas() {
    const {
      directDrawing,
      initialOptionalContent,
      regularAnnotations
    } = this.#useThumbnailCanvas;
    return directDrawing && initialOptionalContent && regularAnnotations ? this.canvas : null;
  }
}

;// ./web/generic_scripting.js

async function docProperties(pdfDocument) {
  const url = "",
    baseUrl = "";
  const {
    info,
    metadata,
    contentDispositionFilename,
    contentLength
  } = await pdfDocument.getMetadata();
  return {
    ...info,
    baseURL: baseUrl,
    filesize: contentLength || (await pdfDocument.getDownloadInfo()).length,
    filename: contentDispositionFilename || getPdfFilenameFromUrl(url),
    metadata: metadata?.getRaw(),
    authors: metadata?.get("dc:creator"),
    numPages: pdfDocument.numPages,
    URL: url
  };
}
class GenericScripting {
  constructor(sandboxBundleSrc, wasmUrl) {
    this._ready = new Promise((resolve, reject) => {
      const sandbox = import(
      /*webpackIgnore: true*/
      /*@vite-ignore*/
      sandboxBundleSrc);
      sandbox.then(pdfjsSandbox => {
        resolve(pdfjsSandbox.QuickJSSandbox(new URL(wasmUrl, location.href).href));
      }).catch(reject);
    });
  }
  async createSandbox(data) {
    const sandbox = await this._ready;
    sandbox.create(data);
  }
  async dispatchEventInSandbox(event) {
    const sandbox = await this._ready;
    setTimeout(() => sandbox.dispatchEvent(event), 0);
  }
  async destroySandbox() {
    const sandbox = await this._ready;
    sandbox.nukeSandbox();
  }
}

;// ./web/pdf_scripting_manager.js













class PDFScriptingManager {
  #closeCapability = null;
  #destroyCapability = null;
  #docProperties = null;
  #eventAC = null;
  #eventBus = null;
  #externalServices = null;
  #objectIds = null;
  #pdfDocument = null;
  #pdfViewer = null;
  #ready = false;
  #scripting = null;
  #willPrintCapability = null;
  constructor({
    eventBus,
    externalServices = null,
    docProperties = null
  }) {
    this.#eventBus = eventBus;
    this.#externalServices = externalServices;
    this.#docProperties = docProperties;
  }
  setViewer(pdfViewer) {
    this.#pdfViewer = pdfViewer;
  }
  async setDocument(pdfDocument) {
    if (this.#pdfDocument) {
      await this.#destroyScripting();
    }
    this.#pdfDocument = pdfDocument;
    if (!pdfDocument) {
      return;
    }
    const [objects, calculationOrder, docActions] = await Promise.all([pdfDocument.getFieldObjects(), pdfDocument.getCalculationOrderIds(), pdfDocument.getJSActions()]);
    if (!objects && !docActions) {
      await this.#destroyScripting();
      return;
    }
    if (pdfDocument !== this.#pdfDocument) {
      return;
    }
    if (objects) {
      this.#objectIds = new Set();
      for (const fields of objects.values()) {
        for (const {
          id
        } of fields) {
          this.#objectIds.add(id);
        }
      }
    }
    try {
      this.#scripting = this.#initScripting();
    } catch (error) {
      console.error("setDocument:", error);
      await this.#destroyScripting();
      return;
    }
    const eventBus = this.#eventBus;
    this.#eventAC = new AbortController();
    const evtOpts = {
      signal: this.#eventAC.signal,
      ...internalOpt
    };
    eventBus.on("updatefromsandbox", event => {
      if (event?.source === window) {
        this.#updateFromSandbox(event.detail);
      }
    }, evtOpts);
    eventBus.on("dispatcheventinsandbox", event => {
      this.#scripting?.dispatchEventInSandbox(event.detail);
    }, evtOpts);
    eventBus.on("pagechanging", ({
      pageNumber,
      previous
    }) => {
      if (pageNumber === previous) {
        return;
      }
      this.#dispatchPageClose(previous);
      this.#dispatchPageOpen(pageNumber);
    }, evtOpts);
    eventBus.on("pagerendered", ({
      pageNumber
    }) => {
      if (!this._pageOpenPending.has(pageNumber)) {
        return;
      }
      if (pageNumber !== this.#pdfViewer.currentPageNumber) {
        return;
      }
      this.#dispatchPageOpen(pageNumber);
    }, evtOpts);
    eventBus.on("pagesdestroy", async () => {
      await this.#dispatchPageClose(this.#pdfViewer.currentPageNumber);
      await this.#scripting?.dispatchEventInSandbox({
        id: "doc",
        name: "WillClose"
      });
      this.#closeCapability?.resolve();
    }, evtOpts);
    try {
      const docProperties = await this.#docProperties(pdfDocument);
      if (pdfDocument !== this.#pdfDocument) {
        return;
      }
      await this.#scripting.createSandbox({
        objects,
        calculationOrder,
        appInfo: {
          platform: navigator.platform,
          language: navigator.language
        },
        docInfo: {
          ...docProperties,
          actions: docActions
        }
      });
      eventBus.dispatch("sandboxcreated", {
        source: this
      });
    } catch (error) {
      console.error("setDocument:", error);
      await this.#destroyScripting();
      return;
    }
    await this.#scripting?.dispatchEventInSandbox({
      id: "doc",
      name: "Open"
    });
    await this.#dispatchPageOpen(this.#pdfViewer.currentPageNumber, true);
    Promise.resolve().then(() => {
      if (pdfDocument === this.#pdfDocument) {
        this.#ready = true;
      }
    });
  }
  async dispatchWillSave() {
    return this.#scripting?.dispatchEventInSandbox({
      id: "doc",
      name: "WillSave"
    });
  }
  async dispatchDidSave() {
    return this.#scripting?.dispatchEventInSandbox({
      id: "doc",
      name: "DidSave"
    });
  }
  async dispatchWillPrint() {
    if (!this.#scripting) {
      return;
    }
    await this.#willPrintCapability?.promise;
    this.#willPrintCapability = Promise.withResolvers();
    try {
      await this.#scripting.dispatchEventInSandbox({
        id: "doc",
        name: "WillPrint"
      });
    } catch (ex) {
      this.#willPrintCapability.resolve();
      this.#willPrintCapability = null;
      throw ex;
    }
    await this.#willPrintCapability.promise;
  }
  async dispatchDidPrint() {
    return this.#scripting?.dispatchEventInSandbox({
      id: "doc",
      name: "DidPrint"
    });
  }
  get destroyPromise() {
    return this.#destroyCapability?.promise || null;
  }
  get ready() {
    return this.#ready;
  }
  get _pageOpenPending() {
    return shadow(this, "_pageOpenPending", new Set());
  }
  get _visitedPages() {
    return shadow(this, "_visitedPages", new Map());
  }
  async #updateFromSandbox(detail) {
    const pdfViewer = this.#pdfViewer;
    const isInPresentationMode = pdfViewer.isInPresentationMode || pdfViewer.isChangingPresentationMode;
    const {
      id,
      siblings,
      command,
      value
    } = detail;
    if (!id) {
      switch (command) {
        case "clear":
          console.clear();
          break;
        case "error":
          console.error(value);
          break;
        case "layout":
          if (!isInPresentationMode) {
            const modes = apiPageLayoutToViewerModes(value);
            pdfViewer.spreadMode = modes.spreadMode;
          }
          break;
        case "page-num":
          pdfViewer.currentPageNumber = value + 1;
          break;
        case "print":
          await pdfViewer.pagesPromise;
          this.#eventBus.dispatch("print", {
            source: this
          });
          break;
        case "println":
          console.log(value);
          break;
        case "zoom":
          if (!isInPresentationMode) {
            pdfViewer.currentScaleValue = value;
          }
          break;
        case "SaveAs":
          this.#eventBus.dispatch("download", {
            source: this
          });
          break;
        case "FirstPage":
          pdfViewer.currentPageNumber = 1;
          break;
        case "LastPage":
          pdfViewer.currentPageNumber = pdfViewer.pagesCount;
          break;
        case "NextPage":
          pdfViewer.nextPage();
          break;
        case "PrevPage":
          pdfViewer.previousPage();
          break;
        case "ZoomViewIn":
          if (!isInPresentationMode) {
            pdfViewer.increaseScale();
          }
          break;
        case "ZoomViewOut":
          if (!isInPresentationMode) {
            pdfViewer.decreaseScale();
          }
          break;
        case "WillPrintFinished":
          this.#willPrintCapability?.resolve();
          this.#willPrintCapability = null;
          break;
      }
      return;
    }
    if (isInPresentationMode && detail.focus) {
      return;
    }
    delete detail.id;
    delete detail.siblings;
    const ids = siblings ? [id, ...siblings] : [id];
    for (const elementId of ids) {
      if (!this.#objectIds?.has(elementId)) {
        continue;
      }
      const element = document.querySelector(`[data-element-id="${elementId}"]`);
      if (element) {
        element.dispatchEvent(new CustomEvent("updatefromsandbox", {
          detail
        }));
      } else {
        this.#pdfDocument?.annotationStorage.setValue(elementId, detail);
      }
    }
  }
  async #dispatchPageOpen(pageNumber, initialize = false) {
    const pdfDocument = this.#pdfDocument,
      visitedPages = this._visitedPages;
    if (initialize) {
      this.#closeCapability = Promise.withResolvers();
    }
    if (!this.#closeCapability) {
      return;
    }
    const pageView = this.#pdfViewer.getPageView(pageNumber - 1);
    if (pageView?.renderingState !== RenderingStates.FINISHED) {
      this._pageOpenPending.add(pageNumber);
      return;
    }
    this._pageOpenPending.delete(pageNumber);
    const actionsPromise = (async () => {
      const actions = await (!visitedPages.has(pageNumber) ? pageView.pdfPage?.getJSActions() : null);
      if (pdfDocument !== this.#pdfDocument) {
        return;
      }
      await this.#scripting?.dispatchEventInSandbox({
        id: "page",
        name: "PageOpen",
        pageNumber,
        actions
      });
    })();
    visitedPages.set(pageNumber, actionsPromise);
  }
  async #dispatchPageClose(pageNumber) {
    const pdfDocument = this.#pdfDocument,
      visitedPages = this._visitedPages;
    if (!this.#closeCapability) {
      return;
    }
    if (this._pageOpenPending.has(pageNumber)) {
      return;
    }
    const actionsPromise = visitedPages.get(pageNumber);
    if (!actionsPromise) {
      return;
    }
    visitedPages.set(pageNumber, null);
    await actionsPromise;
    if (pdfDocument !== this.#pdfDocument) {
      return;
    }
    await this.#scripting?.dispatchEventInSandbox({
      id: "page",
      name: "PageClose",
      pageNumber
    });
  }
  #initScripting() {
    this.#destroyCapability = Promise.withResolvers();
    if (this.#scripting) {
      throw new Error("#initScripting: Scripting already exists.");
    }
    return this.#externalServices.createScripting();
  }
  async #destroyScripting() {
    this.#objectIds = null;
    if (!this.#scripting) {
      this.#pdfDocument = null;
      this.#destroyCapability?.resolve();
      return;
    }
    if (this.#closeCapability) {
      await Promise.race([this.#closeCapability.promise, new Promise(resolve => {
        setTimeout(resolve, 1000);
      })]).catch(() => {});
      this.#closeCapability = null;
    }
    this.#pdfDocument = null;
    try {
      await this.#scripting.destroySandbox();
    } catch {}
    this.#willPrintCapability?.reject(new Error("Scripting destroyed."));
    this.#willPrintCapability = null;
    this.#eventAC?.abort();
    this.#eventAC = null;
    this._pageOpenPending.clear();
    this._visitedPages.clear();
    this.#scripting = null;
    this.#ready = false;
    this.#destroyCapability?.resolve();
  }
}

;// ./web/pdf_scripting_manager.component.js


class PDFScriptingManagerComponents extends PDFScriptingManager {
  constructor(options) {
    if (!options.externalServices) {
      window.addEventListener("updatefromsandbox", event => {
        options.eventBus.dispatch("updatefromsandbox", {
          source: window,
          detail: event.detail
        });
      });
    }
    options.externalServices ||= {
      createScripting: () => new GenericScripting(options.sandboxBundleSrc, options.wasmUrl)
    };
    options.docProperties ||= pdfDocument => docProperties(pdfDocument);
    super(options);
  }
}

// EXTERNAL MODULE: ./node_modules/core-js/modules/es.iterator.every.js
var es_iterator_every = __webpack_require__(1148);
// EXTERNAL MODULE: ./node_modules/core-js/modules/es.iterator.some.js
var es_iterator_some = __webpack_require__(3579);
;// ./web/pdf_rendering_queue.js


const CLEANUP_TIMEOUT = 30000;
class PDFRenderingQueue {
  #highestPriorityPage = null;
  #idleTimeout = null;
  #pdfThumbnailViewer = null;
  #pdfViewer = null;
  isThumbnailViewEnabled = false;
  onIdle = null;
  printing = false;
  constructor() {
    Object.defineProperty(this, "hasViewer", {
      value: () => !!this.#pdfViewer
    });
  }
  setViewer(pdfViewer) {
    this.#pdfViewer = pdfViewer;
  }
  setThumbnailViewer(pdfThumbnailViewer) {
    this.#pdfThumbnailViewer = pdfThumbnailViewer;
  }
  isHighestPriority(view) {
    return this.#highestPriorityPage === view.renderingId;
  }
  renderHighestPriority(currentlyVisiblePages) {
    if (this.#idleTimeout) {
      clearTimeout(this.#idleTimeout);
      this.#idleTimeout = null;
    }
    if (this.#pdfViewer.forceRendering(currentlyVisiblePages)) {
      return;
    }
    if (this.isThumbnailViewEnabled && this.#pdfThumbnailViewer?.forceRendering()) {
      return;
    }
    if (this.printing) {
      return;
    }
    if (this.onIdle) {
      this.#idleTimeout = setTimeout(this.onIdle.bind(this), CLEANUP_TIMEOUT);
    }
  }
  getHighestPriority(visible, views, scrolledDown, preRenderExtra = false, ignoreDetailViews = false) {
    const visibleViews = visible.views,
      numVisible = visibleViews.length;
    if (numVisible === 0) {
      return null;
    }
    for (let i = 0; i < numVisible; i++) {
      const view = visibleViews[i].view;
      if (!this.isViewFinished(view)) {
        return view;
      }
    }
    if (!ignoreDetailViews) {
      for (let i = 0; i < numVisible; i++) {
        const {
          detailView
        } = visibleViews[i].view;
        if (detailView && !this.isViewFinished(detailView)) {
          return detailView;
        }
      }
    }
    const firstId = visible.first.id,
      lastId = visible.last.id;
    if (lastId - firstId + 1 > numVisible) {
      const visibleIds = visible.ids;
      for (let i = 1, ii = lastId - firstId; i < ii; i++) {
        const holeId = scrolledDown ? firstId + i : lastId - i;
        if (visibleIds.has(holeId)) {
          continue;
        }
        const holeView = views[holeId - 1];
        if (!this.isViewFinished(holeView)) {
          return holeView;
        }
      }
    }
    let preRenderIndex = scrolledDown ? lastId : firstId - 2;
    let preRenderView = views[preRenderIndex];
    if (preRenderView && !this.isViewFinished(preRenderView)) {
      return preRenderView;
    }
    if (preRenderExtra) {
      preRenderIndex += scrolledDown ? 1 : -1;
      preRenderView = views[preRenderIndex];
      if (preRenderView && !this.isViewFinished(preRenderView)) {
        return preRenderView;
      }
    }
    return null;
  }
  isViewFinished(view) {
    return view.renderingState === RenderingStates.FINISHED;
  }
  renderView(view) {
    switch (view.renderingState) {
      case RenderingStates.FINISHED:
        return false;
      case RenderingStates.PAUSED:
        this.#highestPriorityPage = view.renderingId;
        view.resume();
        break;
      case RenderingStates.RUNNING:
        this.#highestPriorityPage = view.renderingId;
        break;
      case RenderingStates.INITIAL:
        this.#highestPriorityPage = view.renderingId;
        view.draw().finally(() => {
          this.renderHighestPriority();
        }).catch(reason => {
          if (reason instanceof RenderingCancelledException) {
            return;
          }
          console.error("renderView:", reason);
        });
        break;
    }
    return true;
  }
}

;// ./web/pdf_viewer.js























const DEFAULT_CACHE_SIZE = 10;
const PagesCountLimit = {
  FORCE_SCROLL_MODE_PAGE: 10000,
  FORCE_LAZY_PAGE_INIT: 5000,
  PAUSE_EAGER_PAGE_INIT: 250
};
function isValidAnnotationEditorMode(mode) {
  return Object.values(AnnotationEditorType).includes(mode) && mode !== AnnotationEditorType.DISABLE;
}
class PDFPageViewBuffer {
  #buf = new Set();
  #size = 0;
  constructor(size) {
    this.#size = size;
  }
  push(view) {
    const buf = this.#buf;
    if (buf.has(view)) {
      buf.delete(view);
    }
    buf.add(view);
    if (buf.size > this.#size) {
      this.#destroyFirstView();
    }
  }
  resize(newSize, idsToKeep = null) {
    this.#size = newSize;
    const buf = this.#buf;
    if (idsToKeep) {
      const ii = buf.size;
      let i = 1;
      for (const view of buf) {
        if (idsToKeep.has(view.id)) {
          buf.delete(view);
          buf.add(view);
        }
        if (++i > ii) {
          break;
        }
      }
    }
    while (buf.size > this.#size) {
      this.#destroyFirstView();
    }
  }
  has(view) {
    return this.#buf.has(view);
  }
  [Symbol.iterator]() {
    return this.#buf.keys();
  }
  #destroyFirstView() {
    const firstView = this.#buf.keys().next().value;
    firstView?.destroy();
    this.#buf.delete(firstView);
  }
}
class PDFViewer {
  #buffer = null;
  #altTextManager = null;
  #annotationEditorHighlightColors = null;
  #annotationEditorMode = AnnotationEditorType.NONE;
  #annotationEditorUIManager = null;
  #annotationMode = AnnotationMode.ENABLE_FORMS;
  #commentManager = null;
  #containerTopLeft = null;
  #editorUndoBar = null;
  #enableHighlightFloatingButton = false;
  #enablePermissions = false;
  #enableUpdatedAddImage = false;
  #enableNewAltTextWhenAddingImage = false;
  #enableAutoLinking = true;
  #abortSignal = null;
  #eventAC = null;
  #minDurationToUpdateCanvas = 0;
  #mlManager = null;
  #panPosition = [NaN, NaN];
  #printingAllowed = true;
  #scrollTimeoutId = null;
  #staleLocation = false;
  #switchAnnotationEditorModeAC = null;
  #switchAnnotationEditorModeTimeoutId = null;
  #copyAllInProgress = false;
  #hiddenCopyElement = null;
  #previousContainerHeight = 0;
  #resizeObserver = new ResizeObserver(this.#resizeObserverCallback.bind(this));
  #scrollModePageState = null;
  #scaleTimeoutId = null;
  #signatureManager = null;
  #supportsPinchToZoom = true;
  #textLayerMode = TextLayerMode.ENABLE;
  #viewerAlert = null;
  #copiedPageViews = null;
  #savedPageViews = null;
  #deletedPageNumbers = null;
  constructor(options) {
    const viewerVersion = "6.3.289";
    if (version !== viewerVersion) {
      throw new Error(`The API version "${version}" does not match the Viewer version "${viewerVersion}".`);
    }
    this.container = options.container;
    this.viewer = options.viewer || options.container.firstElementChild;
    this.#viewerAlert = options.viewerAlert || null;
    if (this.container?.tagName !== "DIV" || this.viewer?.tagName !== "DIV") {
      throw new Error("Invalid `container` and/or `viewer` option.");
    }
    if (this.container.offsetParent && getComputedStyle(this.container).position !== "absolute") {
      throw new Error("The `container` must be absolutely positioned.");
    }
    this.#resizeObserver.observe(this.container);
    this.eventBus = options.eventBus;
    this.linkService = options.linkService || new SimpleLinkService();
    this.downloadManager = options.downloadManager || null;
    this.findController = options.findController || null;
    this.#altTextManager = options.altTextManager || null;
    this.#commentManager = options.commentManager || null;
    this.#signatureManager = options.signatureManager || null;
    this.#editorUndoBar = options.editorUndoBar || null;
    if (this.findController) {
      this.findController.onIsPageVisible = pageNumber => this._getVisiblePages().ids.has(pageNumber);
    }
    this._scriptingManager = options.scriptingManager || null;
    this.#textLayerMode = options.textLayerMode ?? TextLayerMode.ENABLE;
    this.#annotationMode = options.annotationMode ?? AnnotationMode.ENABLE_FORMS;
    this.#annotationEditorMode = options.annotationEditorMode ?? AnnotationEditorType.NONE;
    this.#annotationEditorHighlightColors = options.annotationEditorHighlightColors || null;
    this.#enableHighlightFloatingButton = options.enableHighlightFloatingButton === true;
    this.#enableUpdatedAddImage = options.enableUpdatedAddImage === true;
    this.#enableNewAltTextWhenAddingImage = options.enableNewAltTextWhenAddingImage === true;
    this.imageResourcesPath = options.imageResourcesPath || "";
    this.enablePrintAutoRotate = options.enablePrintAutoRotate || false;
    this.removePageBorders = options.removePageBorders || false;
    this.maxCanvasPixels = options.maxCanvasPixels;
    this.maxCanvasDim = options.maxCanvasDim;
    this.capCanvasAreaFactor = options.capCanvasAreaFactor;
    this.enableDetailCanvas = options.enableDetailCanvas ?? true;
    this.enableOptimizedPartialRendering = options.enableOptimizedPartialRendering ?? false;
    this.enableSelectionRendering = options.enableSelectionRendering !== false;
    this.imagesRightClickMinSize = options.imagesRightClickMinSize ?? -1;
    this.l10n = options.l10n;
    this.l10n ||= new genericl10n_GenericL10n();
    this.#enablePermissions = options.enablePermissions || false;
    this.pageColors = options.pageColors || null;
    this.#mlManager = options.mlManager || null;
    this.#supportsPinchToZoom = options.supportsPinchToZoom !== false;
    this.#enableAutoLinking = options.enableAutoLinking !== false;
    this.#minDurationToUpdateCanvas = options.minDurationToUpdateCanvas ?? 500;
    this.defaultRenderingQueue = !options.renderingQueue;
    if (this.defaultRenderingQueue) {
      this.renderingQueue = new PDFRenderingQueue();
      this.renderingQueue.setViewer(this);
    } else {
      this.renderingQueue = options.renderingQueue;
    }
    const {
      abortSignal
    } = options;
    this.#abortSignal = abortSignal || null;
    abortSignal?.addEventListener("abort", () => {
      this.#resizeObserver.disconnect();
      this.#resizeObserver = null;
    }, {
      once: true
    });
    this.scroll = watchScroll(this.container, this._scrollUpdate.bind(this), abortSignal);
    this.presentationModeState = PresentationModeState.UNKNOWN;
    this._resetView();
    if (this.removePageBorders) {
      this.viewer.classList.add("removePageBorders");
    }
    this.#updateContainerHeightCss();
    this.eventBus.on("thumbnailrendered", ({
      pageNumber,
      pdfPage
    }) => {
      const pageView = this._pages[pageNumber - 1];
      if (!this.#buffer.has(pageView)) {
        pdfPage?.cleanup();
      }
    }, internalOpt);
    if (!options.l10n) {
      this.l10n.translate(this.container);
    }
  }
  get printingAllowed() {
    return this.#printingAllowed;
  }
  get pagesCount() {
    return this._pages.length;
  }
  getPageView(index) {
    return this._pages[index];
  }
  getCachedPageViews() {
    return new Set(this.#buffer);
  }
  get pageViewsReady() {
    return this._pages.every(pageView => pageView?.pdfPage);
  }
  clearSelection() {
    const selection = document.getSelection();
    if (!selection || selection.isCollapsed) {
      return;
    }
    for (let i = 0, ii = selection.rangeCount; i < ii; i++) {
      if (selection.getRangeAt(i).intersectsNode(this.viewer)) {
        selection.removeAllRanges?.();
        selection.empty?.();
        return;
      }
    }
  }
  get renderForms() {
    return this.#annotationMode === AnnotationMode.ENABLE_FORMS;
  }
  get enableScripting() {
    return !!this._scriptingManager;
  }
  get currentPageNumber() {
    return this._currentPageNumber;
  }
  set currentPageNumber(val) {
    if (!Number.isInteger(val)) {
      throw new Error("Invalid page number.");
    }
    if (!this.pdfDocument) {
      return;
    }
    if (this._currentPageNumber !== val) {
      this.clearSelection();
    }
    if (!this._setCurrentPageNumber(val, true)) {
      console.error(`currentPageNumber: "${val}" is not a valid page.`);
    }
  }
  _setCurrentPageNumber(val, resetCurrentPageView = false) {
    if (this._currentPageNumber === val) {
      if (resetCurrentPageView) {
        this.#resetCurrentPageView();
      }
      return true;
    }
    if (!(0 < val && val <= this.pagesCount)) {
      return false;
    }
    const previous = this._currentPageNumber;
    this._currentPageNumber = val;
    this.eventBus.dispatch("pagechanging", {
      source: this,
      pageNumber: val,
      pageLabel: this._pageLabels?.[val - 1] ?? null,
      previous
    });
    if (resetCurrentPageView) {
      this.#resetCurrentPageView();
    }
    return true;
  }
  get currentPageLabel() {
    return this._pageLabels?.[this._currentPageNumber - 1] ?? null;
  }
  set currentPageLabel(val) {
    if (!this.pdfDocument) {
      return;
    }
    let page = val | 0;
    if (this._pageLabels) {
      const i = this._pageLabels.indexOf(val);
      if (i >= 0) {
        page = i + 1;
      }
    }
    if (this._currentPageNumber !== page) {
      this.clearSelection();
    }
    if (!this._setCurrentPageNumber(page, true)) {
      console.error(`currentPageLabel: "${val}" is not a valid page.`);
    }
  }
  get currentScale() {
    return this._currentScale !== (/* inlined export .UNKNOWN_SCALE */0) ? this._currentScale : (/* inlined export .DEFAULT_SCALE */1);
  }
  set currentScale(val) {
    if (isNaN(val)) {
      throw new Error("Invalid numeric scale.");
    }
    if (!this.pdfDocument) {
      return;
    }
    this.#setScale(val, {
      noScroll: false
    });
  }
  get currentScaleValue() {
    return this._currentScaleValue;
  }
  set currentScaleValue(val) {
    if (!this.pdfDocument) {
      return;
    }
    this.#setScale(val, {
      noScroll: false
    });
  }
  get pagesRotation() {
    return this._pagesRotation;
  }
  set pagesRotation(rotation) {
    if (!isValidRotation(rotation)) {
      throw new Error("Invalid pages rotation angle.");
    }
    if (!this.pdfDocument) {
      return;
    }
    rotation %= 360;
    if (rotation < 0) {
      rotation += 360;
    }
    if (this._pagesRotation === rotation) {
      return;
    }
    this.clearSelection();
    this._pagesRotation = rotation;
    const pageNumber = this._currentPageNumber;
    this.refresh(true, {
      rotation
    });
    if (this._currentScaleValue) {
      this.#setScale(this._currentScaleValue, {
        noScroll: true
      });
    }
    this.eventBus.dispatch("rotationchanging", {
      source: this,
      pagesRotation: rotation,
      pageNumber
    });
    if (this.defaultRenderingQueue) {
      this.update();
    }
  }
  get firstPagePromise() {
    return this.pdfDocument ? this._firstPageCapability.promise : null;
  }
  get onePageRendered() {
    return this.pdfDocument ? this._onePageRenderedCapability.promise : null;
  }
  get pagesPromise() {
    return this.pdfDocument ? this._pagesCapability.promise : null;
  }
  get _layerProperties() {
    const self = this;
    return shadow(this, "_layerProperties", {
      get annotationEditorUIManager() {
        return self.#annotationEditorUIManager;
      },
      get annotationStorage() {
        return self.pdfDocument?.annotationStorage;
      },
      get downloadManager() {
        return self.downloadManager;
      },
      get enableComment() {
        return !!self.#commentManager;
      },
      get enableScripting() {
        return !!self._scriptingManager;
      },
      get fieldObjectsPromise() {
        return self.pdfDocument?.getFieldObjects();
      },
      get findController() {
        return self.findController;
      },
      get hasJSActionsPromise() {
        return self.pdfDocument?.hasJSActions();
      },
      get linkService() {
        return self.linkService;
      }
    });
  }
  #setPrintingAllowed(isAllowed) {
    this.#printingAllowed = isAllowed;
    this.eventBus.dispatch("printingallowed", {
      source: this,
      isAllowed
    });
  }
  #initializePermissions(permissions) {
    const params = {
      annotationEditorMode: this.#annotationEditorMode,
      annotationMode: this.#annotationMode,
      textLayerMode: this.#textLayerMode
    };
    if (!permissions) {
      this.#setPrintingAllowed(true);
      return params;
    }
    this.#setPrintingAllowed(permissions.has(PermissionFlag.PRINT_HIGH_QUALITY) || permissions.has(PermissionFlag.PRINT));
    if (!permissions.has(PermissionFlag.COPY) && this.#textLayerMode === TextLayerMode.ENABLE) {
      params.textLayerMode = TextLayerMode.ENABLE_PERMISSIONS;
    }
    if (!permissions.has(PermissionFlag.MODIFY_CONTENTS)) {
      params.annotationEditorMode = AnnotationEditorType.DISABLE;
    }
    if (!permissions.has(PermissionFlag.MODIFY_ANNOTATIONS) && !permissions.has(PermissionFlag.FILL_INTERACTIVE_FORMS) && this.#annotationMode === AnnotationMode.ENABLE_FORMS) {
      params.annotationMode = AnnotationMode.ENABLE;
    }
    return params;
  }
  async #onePageRenderedOrForceFetch(signal) {
    if (document.visibilityState === "hidden" || !this.container.offsetParent || this._getVisiblePages().views.length === 0) {
      return;
    }
    const hiddenCapability = Promise.withResolvers(),
      ac = new AbortController();
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "hidden") {
        hiddenCapability.resolve();
      }
    }, {
      signal: AbortSignal.any([signal, ac.signal])
    });
    await Promise.race([this._onePageRenderedCapability.promise, hiddenCapability.promise]);
    ac.abort();
  }
  async getAllText(interruptSignal = null) {
    const texts = [];
    const buffer = [];
    for (let pageNum = 1, pagesCount = this.pdfDocument.numPages; pageNum <= pagesCount; ++pageNum) {
      if (interruptSignal?.aborted) {
        return null;
      }
      buffer.length = 0;
      const page = await this.pdfDocument.getPage(pageNum);
      const {
        items
      } = await page.getTextContent();
      for (const item of items) {
        if (item.str) {
          buffer.push(item.str);
        }
        if (item.hasEOL) {
          buffer.push("\n");
        }
      }
      texts.push(removeNullCharacters(buffer.join("")));
    }
    return texts.join("\n");
  }
  #copyCallback(textLayerMode, event) {
    const selection = document.getSelection();
    const {
      focusNode,
      anchorNode
    } = selection;
    if (anchorNode && focusNode && selection.containsNode(this.#hiddenCopyElement)) {
      if (this.#copyAllInProgress || textLayerMode === TextLayerMode.ENABLE_PERMISSIONS) {
        stopEvent(event);
        return;
      }
      this.#copyAllInProgress = true;
      const {
        classList
      } = this.viewer;
      classList.add("copyAll");
      const keydownAC = new AbortController(),
        interruptAC = new AbortController();
      window.addEventListener("keydown", ev => {
        if (ev.key === "Escape") {
          interruptAC.abort();
        }
      }, {
        signal: keydownAC.signal
      });
      this.getAllText(interruptAC.signal).then(async text => {
        if (text !== null) {
          await navigator.clipboard.writeText(text);
        }
      }).catch(reason => {
        console.warn(`Something goes wrong when extracting the text: ${reason.message}`);
      }).finally(() => {
        this.#copyAllInProgress = false;
        keydownAC.abort();
        classList.remove("copyAll");
      });
      stopEvent(event);
    }
  }
  setDocument(pdfDocument) {
    if (this.pdfDocument) {
      this.eventBus.dispatch("pagesdestroy", {
        source: this
      });
      this._cancelRendering();
      this._resetView();
      this.findController?.setDocument(null);
      this._scriptingManager?.setDocument(null);
      this.#annotationEditorUIManager?.destroy();
      this.#annotationEditorUIManager = null;
      this.#annotationEditorMode = AnnotationEditorType.NONE;
      this.#printingAllowed = true;
    }
    this.pdfDocument = pdfDocument;
    if (!pdfDocument) {
      return;
    }
    const pagesCount = pdfDocument.numPages;
    const firstPagePromise = pdfDocument.getPage(1);
    const optionalContentConfigPromise = pdfDocument.getOptionalContentConfig({
      intent: "display"
    });
    const permissionsPromise = this.#enablePermissions ? pdfDocument.getPermissions() : Promise.resolve();
    const {
      eventBus,
      pageColors,
      viewer
    } = this;
    this.#eventAC = new AbortController();
    const {
      signal
    } = this.#eventAC;
    const evtOpts = {
      signal,
      ...internalOpt
    };
    if (pagesCount > PagesCountLimit.FORCE_SCROLL_MODE_PAGE) {
      console.warn("Forcing PAGE-scrolling for performance reasons, given the length of the document.");
      const mode = this._scrollMode = ScrollMode.PAGE;
      eventBus.dispatch("scrollmodechanged", {
        source: this,
        mode
      });
    }
    this._pagesCapability.promise.then(() => {
      eventBus.dispatch("pagesloaded", {
        source: this,
        pagesCount
      });
    }, () => {});
    const onBeforeDraw = evt => {
      const pageView = this._pages[evt.pageNumber - 1];
      if (!pageView) {
        return;
      }
      this.#buffer.push(pageView);
    };
    eventBus.on("pagerender", onBeforeDraw, evtOpts);
    const onAfterDraw = evt => {
      if (evt.cssTransform || evt.isDetailView) {
        return;
      }
      this._onePageRenderedCapability.resolve({
        timestamp: evt.timestamp
      });
      eventBus.off("pagerendered", onAfterDraw);
    };
    eventBus.on("pagerendered", onAfterDraw, evtOpts);
    Promise.all([firstPagePromise, permissionsPromise]).then(([firstPdfPage, permissions]) => {
      if (pdfDocument !== this.pdfDocument) {
        return;
      }
      this._firstPageCapability.resolve(firstPdfPage);
      this._optionalContentConfigPromise = optionalContentConfigPromise;
      const {
        annotationEditorMode,
        annotationMode,
        textLayerMode
      } = this.#initializePermissions(permissions);
      if (textLayerMode !== TextLayerMode.DISABLE) {
        const element = this.#hiddenCopyElement = document.createElement("div");
        element.id = "hiddenCopyElement";
        element.style.cssText = "position:absolute;top:0;left:0;width:0;height:0;display:none";
        viewer.before(element);
      }
      if (annotationEditorMode !== AnnotationEditorType.DISABLE) {
        const mode = annotationEditorMode;
        if (pdfDocument.isPureXfa) {
          console.warn("Warning: XFA-editing is not implemented.");
        } else if (isValidAnnotationEditorMode(mode)) {
          this.#annotationEditorUIManager = new AnnotationEditorUIManager(this.container, viewer, this.#viewerAlert, this.#altTextManager, this.#commentManager, this.#signatureManager, eventBus, pdfDocument, pageColors, this.#annotationEditorHighlightColors, this.#enableHighlightFloatingButton, this.#enableUpdatedAddImage, this.#enableNewAltTextWhenAddingImage, this.#mlManager, this.#editorUndoBar, this.#supportsPinchToZoom);
          eventBus.dispatch("annotationeditoruimanager", {
            source: this,
            uiManager: this.#annotationEditorUIManager
          });
          if (mode !== AnnotationEditorType.NONE) {
            this.#preloadEditingData(mode);
            this.#annotationEditorUIManager.updateMode(mode);
          }
        } else {
          console.error(`Invalid AnnotationEditor mode: ${mode}`);
        }
      }
      const viewerElement = this._scrollMode === ScrollMode.PAGE ? null : viewer;
      const scale = this.currentScale;
      const viewport = firstPdfPage.getViewport({
        scale: scale * PixelsPerInch.PDF_TO_CSS_UNITS
      });
      viewer.style.setProperty("--scale-factor", viewport.scale);
      if (pageColors?.background) {
        viewer.style.setProperty("--page-bg-color", pageColors.background);
      }
      if (pageColors?.foreground === "CanvasText" || pageColors?.background === "Canvas") {
        viewer.style.setProperty("--hcm-highlight-filter", pdfDocument.filterFactory.addHighlightHCMFilter("highlight", "CanvasText", "Canvas", "HighlightText", "Highlight"));
        viewer.style.setProperty("--hcm-highlight-selected-filter", pdfDocument.filterFactory.addHighlightHCMFilter("highlight_selected", "CanvasText", "Canvas", "HighlightText", "ButtonText"));
      }
      for (let pageNum = 1; pageNum <= pagesCount; ++pageNum) {
        const pageView = new PDFPageView({
          container: viewerElement,
          eventBus,
          id: pageNum,
          scale,
          defaultViewport: viewport.clone(),
          optionalContentConfigPromise,
          renderingQueue: this.renderingQueue,
          textLayerMode,
          annotationMode,
          imageResourcesPath: this.imageResourcesPath,
          maxCanvasPixels: this.maxCanvasPixels,
          maxCanvasDim: this.maxCanvasDim,
          capCanvasAreaFactor: this.capCanvasAreaFactor,
          enableDetailCanvas: this.enableDetailCanvas,
          enableOptimizedPartialRendering: this.enableOptimizedPartialRendering,
          enableSelectionRendering: this.enableSelectionRendering,
          imagesRightClickMinSize: this.imagesRightClickMinSize,
          pageColors,
          l10n: this.l10n,
          layerProperties: this._layerProperties,
          enableAutoLinking: this.#enableAutoLinking,
          minDurationToUpdateCanvas: this.#minDurationToUpdateCanvas,
          commentManager: this.#commentManager,
          abortSignal: this.#abortSignal
        });
        this._pages.push(pageView);
      }
      this._pages[0]?.setPdfPage(firstPdfPage);
      if (this._scrollMode === ScrollMode.PAGE) {
        this.#ensurePageViewVisible();
      } else if (this._spreadMode !== SpreadMode.NONE) {
        this._updateSpreadMode();
      }
      eventBus.on("annotationeditorlayerrendered", evt => {
        if (this.#annotationEditorUIManager) {
          eventBus.dispatch("annotationeditormodechanged", {
            source: this,
            mode: this.#annotationEditorMode
          });
        }
      }, {
        once: true,
        signal,
        ...internalOpt
      });
      this.#onePageRenderedOrForceFetch(signal).then(async () => {
        if (pdfDocument !== this.pdfDocument) {
          return;
        }
        this.findController?.setDocument(pdfDocument);
        this._scriptingManager?.setDocument(pdfDocument);
        if (this.#hiddenCopyElement) {
          document.addEventListener("copy", this.#copyCallback.bind(this, textLayerMode), {
            signal
          });
        }
        if (pdfDocument.loadingParams.disableAutoFetch || pagesCount > PagesCountLimit.FORCE_LAZY_PAGE_INIT) {
          this._pagesCapability.resolve();
          return;
        }
        let getPagesLeft = pagesCount - 1;
        if (getPagesLeft <= 0) {
          this._pagesCapability.resolve();
          return;
        }
        for (let pageNum = 2; pageNum <= pagesCount; ++pageNum) {
          const promise = pdfDocument.getPage(pageNum).then(pdfPage => {
            const pageView = this._pages[pageNum - 1];
            if (!pageView.pdfPage) {
              pageView.setPdfPage(pdfPage);
            }
            if (--getPagesLeft === 0) {
              this._pagesCapability.resolve();
            }
          }, reason => {
            console.error(`Unable to get page ${pageNum} to initialize viewer`, reason);
            if (--getPagesLeft === 0) {
              this._pagesCapability.resolve();
            }
          });
          if (pageNum % PagesCountLimit.PAUSE_EAGER_PAGE_INIT === 0) {
            await promise;
          }
        }
      });
      eventBus.dispatch("pagesinit", {
        source: this
      });
      pdfDocument.getMetadata().then(({
        info
      }) => {
        if (pdfDocument !== this.pdfDocument) {
          return;
        }
        if (info.Language) {
          viewer.lang = info.Language;
        }
      });
      if (this.defaultRenderingQueue) {
        this.update();
      }
    }).catch(reason => {
      console.error("Unable to initialize viewer", reason);
      this._pagesCapability.reject(reason);
    });
  }
  onPagesEdited({
    pagesMapper,
    type,
    hasBeenCut,
    pageNumbers
  }) {
    if (type === "copy") {
      this.#copiedPageViews = new Map();
      for (const pageNum of pageNumbers) {
        this.#copiedPageViews.set(pageNum, this._pages[pageNum - 1]);
      }
      return;
    }
    if (type === "cancelCopy") {
      this.#copiedPageViews = null;
      return;
    }
    const isCut = type === "cut";
    if (isCut || type === "delete") {
      this.#savedPageViews = this._pages;
      this.#deletedPageNumbers = pageNumbers;
    }
    if (type === "cancelDelete") {
      this.#deletedPageNumbers = null;
      if (!this.#savedPageViews) {
        return;
      }
      const viewerElement = this._scrollMode === ScrollMode.PAGE ? null : this.viewer;
      if (viewerElement) {
        this.#annotationEditorUIManager?.startUpdatePages();
        const fragment = document.createDocumentFragment();
        for (let i = 0, ii = this.#savedPageViews.length; i < ii; i++) {
          const page = this.#savedPageViews[i];
          page.updatePageNumber(i + 1);
          fragment.append(page.div);
        }
        viewerElement.replaceChildren(fragment);
        this.#annotationEditorUIManager?.endUpdatePages();
      }
      this._pages = this.#savedPageViews;
      this.#savedPageViews = null;
      return;
    }
    if (type === "cleanSavedData") {
      if (this.#deletedPageNumbers) {
        if (this.#savedPageViews) {
          for (const pageNum of this.#deletedPageNumbers) {
            this.#savedPageViews[pageNum - 1].deleteMe();
          }
          this.#savedPageViews = null;
        }
        this.#deletedPageNumbers = null;
      }
      return;
    }
    this._currentPageNumber = 0;
    const prevPages = this._pages;
    const newPages = this._pages = [];
    this.#annotationEditorUIManager?.startUpdatePages();
    for (let i = 1, ii = pagesMapper.pagesNumber; i <= ii; i++) {
      const prevPageNumber = pagesMapper.getPrevPageNumber(i);
      if (prevPageNumber < 0) {
        let page = this.#copiedPageViews.get(-prevPageNumber);
        if (hasBeenCut) {
          page.updatePageNumber(i);
        } else {
          this.#annotationEditorUIManager?.clonePage(-prevPageNumber - 1, i - 1);
          page = page.clone(i);
        }
        newPages.push(page);
        continue;
      }
      const page = prevPages[prevPageNumber - 1];
      newPages.push(page);
      page.updatePageNumber(i);
    }
    this.#annotationEditorUIManager?.endUpdatePages();
    if (type === "paste") {
      this.#copiedPageViews = null;
    }
    const viewerElement = this._scrollMode === ScrollMode.PAGE ? null : this.viewer;
    if (viewerElement) {
      const fragment = document.createDocumentFragment();
      for (const {
        div
      } of newPages) {
        fragment.append(div);
      }
      viewerElement.replaceChildren(fragment);
    }
    setTimeout(() => {
      this.forceRendering();
    });
  }
  setPageLabels(labels) {
    if (!this.pdfDocument) {
      return;
    }
    if (!labels) {
      this._pageLabels = null;
    } else if (!(Array.isArray(labels) && this.pdfDocument.numPages === labels.length)) {
      this._pageLabels = null;
      console.error(`setPageLabels: Invalid page labels.`);
    } else {
      this._pageLabels = labels;
    }
    for (let i = 0, ii = this._pages.length; i < ii; i++) {
      this._pages[i].setPageLabel(this._pageLabels?.[i] ?? null);
    }
  }
  _resetView() {
    this._pages = [];
    this._currentPageNumber = 1;
    this._currentScale = (/* inlined export .UNKNOWN_SCALE */0);
    this._currentScaleValue = null;
    this._pageLabels = null;
    this.#buffer = new PDFPageViewBuffer(DEFAULT_CACHE_SIZE);
    this._location = null;
    this._pagesRotation = 0;
    this._optionalContentConfigPromise = null;
    this._firstPageCapability = Promise.withResolvers();
    this._onePageRenderedCapability = Promise.withResolvers();
    this._pagesCapability = Promise.withResolvers();
    this._scrollMode = ScrollMode.VERTICAL;
    this._previousScrollMode = ScrollMode.UNKNOWN;
    this._spreadMode = SpreadMode.NONE;
    this.#scrollModePageState = {
      previousPageNumber: 1,
      scrollDown: true,
      pages: []
    };
    this.#eventAC?.abort();
    this.#eventAC = null;
    this.viewer.textContent = "";
    this._updateScrollMode();
    this.viewer.removeAttribute("lang");
    this.#hiddenCopyElement?.remove();
    this.#hiddenCopyElement = null;
    this.#cleanupTimeouts();
    this.#cleanupSwitchAnnotationEditorMode();
  }
  #ensurePageViewVisible() {
    if (this._scrollMode !== ScrollMode.PAGE) {
      throw new Error("#ensurePageViewVisible: Invalid scrollMode value.");
    }
    const pageNumber = this._currentPageNumber,
      state = this.#scrollModePageState,
      viewer = this.viewer;
    viewer.textContent = "";
    state.pages.length = 0;
    if (this._spreadMode === SpreadMode.NONE && !this.isInPresentationMode) {
      const pageView = this._pages[pageNumber - 1];
      viewer.append(pageView.div);
      state.pages.push(pageView);
    } else {
      const pageIndexSet = new Set(),
        parity = this._spreadMode - 1;
      if (parity === -1) {
        pageIndexSet.add(pageNumber - 1);
      } else if (pageNumber % 2 !== parity) {
        pageIndexSet.add(pageNumber - 1);
        pageIndexSet.add(pageNumber);
      } else {
        pageIndexSet.add(pageNumber - 2);
        pageIndexSet.add(pageNumber - 1);
      }
      const spread = document.createElement("div");
      spread.className = "spread";
      if (this.isInPresentationMode) {
        const dummyPage = document.createElement("div");
        dummyPage.className = "dummyPage";
        spread.append(dummyPage);
      }
      for (const i of pageIndexSet) {
        const pageView = this._pages[i];
        if (!pageView) {
          continue;
        }
        spread.append(pageView.div);
        state.pages.push(pageView);
      }
      viewer.append(spread);
    }
    state.scrollDown = pageNumber >= state.previousPageNumber;
    state.previousPageNumber = pageNumber;
  }
  _scrollUpdate() {
    if (this.pagesCount === 0) {
      return;
    }
    if (this.#scrollTimeoutId) {
      clearTimeout(this.#scrollTimeoutId);
    }
    this.#scrollTimeoutId = setTimeout(() => {
      this.#scrollTimeoutId = null;
      this.update();
    }, 100);
    this.update();
  }
  #scrollIntoView(pageView, pageSpot = null) {
    const {
      div,
      id
    } = pageView;
    if (this._currentPageNumber !== id) {
      this._setCurrentPageNumber(id);
    }
    if (this._scrollMode === ScrollMode.PAGE) {
      this.#ensurePageViewVisible();
      this.update();
    }
    if (!pageSpot && !this.isInPresentationMode) {
      const left = div.offsetLeft + div.clientLeft,
        right = left + div.clientWidth;
      const {
        scrollLeft,
        clientWidth
      } = this.container;
      if (this._scrollMode === ScrollMode.HORIZONTAL || left < scrollLeft || right > scrollLeft + clientWidth) {
        pageSpot = {
          left: 0,
          top: 0
        };
      }
    }
    scrollIntoView(div, pageSpot);
    if (!this._currentScaleValue && this._location) {
      this._location = null;
    }
  }
  #isSameScale(newScale) {
    return newScale === this._currentScale || Math.abs(newScale - this._currentScale) < 1e-15;
  }
  panBy(dx, dy) {
    const {
      container
    } = this;
    const position = this.#panPosition;
    const {
      scrollLeft,
      scrollTop
    } = container;
    const left = (Math.abs(scrollLeft - position[0]) < 1 ? position[0] : scrollLeft) - dx;
    const top = (Math.abs(scrollTop - position[1]) < 1 ? position[1] : scrollTop) - dy;
    position[0] = left;
    position[1] = top;
    container.scrollLeft = left;
    container.scrollTop = top;
    this.#staleLocation = true;
  }
  #refreshLocation() {
    if (!this.#staleLocation) {
      return;
    }
    const {
      first
    } = this._getVisiblePages();
    if (first) {
      this._updateLocation(first);
    }
  }
  #setScaleUpdatePages(newScale, newValue, {
    noScroll = false,
    preset = false,
    drawingDelay = -1,
    origin = null,
    pan = null
  }) {
    this._currentScaleValue = newValue.toString();
    if (this.#isSameScale(newScale)) {
      if (pan && !noScroll) {
        this.panBy(pan[0], pan[1]);
      }
      if (preset) {
        this.eventBus.dispatch("scalechanging", {
          source: this,
          scale: newScale,
          presetValue: newValue
        });
      }
      return;
    }
    this.clearSelection();
    this.viewer.style.setProperty("--scale-factor", newScale * PixelsPerInch.PDF_TO_CSS_UNITS);
    const postponeDrawing = drawingDelay >= 0 && drawingDelay < 1000;
    this.refresh(true, {
      scale: newScale,
      drawingDelay: postponeDrawing ? drawingDelay : -1
    });
    if (postponeDrawing) {
      this.#scaleTimeoutId = setTimeout(() => {
        this.#scaleTimeoutId = null;
        this.refresh();
      }, drawingDelay);
    }
    const previousScale = this._currentScale;
    this._currentScale = newScale;
    if (!noScroll) {
      this.#refreshLocation();
      let page = this._currentPageNumber,
        dest;
      if (this._location && !(this.isInPresentationMode || this.isChangingPresentationMode)) {
        page = this._location.pageNumber;
        dest = [null, {
          name: "XYZ"
        }, this._location.left, this._location.top, null];
      }
      this.scrollPageIntoView({
        pageNumber: page,
        destArray: dest,
        allowNegativeOffset: true
      });
      let dx = pan?.[0] ?? 0,
        dy = pan?.[1] ?? 0;
      if (Array.isArray(origin)) {
        const scaleDiff = newScale / previousScale - 1;
        const [top, left] = this.containerTopLeft;
        dx -= (origin[0] - left) * scaleDiff;
        dy -= (origin[1] - top) * scaleDiff;
      }
      if (dx || dy) {
        this.panBy(dx, dy);
      }
    }
    this.eventBus.dispatch("scalechanging", {
      source: this,
      scale: newScale,
      presetValue: preset ? newValue : undefined
    });
    if (this.defaultRenderingQueue) {
      this.update();
    }
  }
  get #pageWidthScaleFactor() {
    return this._spreadMode !== SpreadMode.NONE && this._scrollMode !== ScrollMode.HORIZONTAL ? 2 : 1;
  }
  #setScale(value, options) {
    let scale = parseFloat(value);
    if (scale > 0) {
      options.preset = false;
      this.#setScaleUpdatePages(scale, value, options);
    } else {
      const currentPage = this._pages[this._currentPageNumber - 1];
      if (!currentPage) {
        return;
      }
      let hPadding = (/* inlined export .SCROLLBAR_PADDING */40),
        vPadding = (/* inlined export .VERTICAL_PADDING */5);
      if (this.isInPresentationMode) {
        hPadding = vPadding = 4;
        if (this._spreadMode !== SpreadMode.NONE) {
          hPadding *= 2;
        }
      } else if (this.removePageBorders) {
        hPadding = vPadding = 0;
      } else if (this._scrollMode === ScrollMode.HORIZONTAL) {
        [hPadding, vPadding] = [vPadding, hPadding];
      }
      const pageWidthScale = (this.container.clientWidth - hPadding) / currentPage.width * currentPage.scale / this.#pageWidthScaleFactor;
      const pageHeightScale = (this.container.clientHeight - vPadding) / currentPage.height * currentPage.scale;
      switch (value) {
        case "page-actual":
          scale = 1;
          break;
        case "page-width":
          scale = pageWidthScale;
          break;
        case "page-height":
          scale = pageHeightScale;
          break;
        case "page-fit":
          scale = Math.min(pageWidthScale, pageHeightScale);
          break;
        case "auto":
          const horizontalScale = isPortraitOrientation(currentPage) ? pageWidthScale : Math.min(pageHeightScale, pageWidthScale);
          scale = Math.min((/* inlined export .MAX_AUTO_SCALE */1.25), horizontalScale);
          break;
        default:
          console.error(`#setScale: "${value}" is an unknown zoom value.`);
          return;
      }
      options.preset = true;
      this.#setScaleUpdatePages(scale, value, options);
    }
  }
  #resetCurrentPageView() {
    const pageView = this._pages[this._currentPageNumber - 1];
    if (this.isInPresentationMode) {
      this.#setScale(this._currentScaleValue, {
        noScroll: true
      });
    }
    this.#scrollIntoView(pageView);
  }
  pageLabelToPageNumber(label) {
    if (!this._pageLabels) {
      return null;
    }
    const i = this._pageLabels.indexOf(label);
    if (i < 0) {
      return null;
    }
    return i + 1;
  }
  scrollPageIntoView({
    pageNumber,
    destArray = null,
    allowNegativeOffset = false,
    ignoreDestinationZoom = false,
    center = null
  }) {
    if (!this.pdfDocument) {
      return;
    }
    const pageView = Number.isInteger(pageNumber) && this._pages[pageNumber - 1];
    if (!pageView) {
      console.error(`scrollPageIntoView: "${pageNumber}" is not a valid pageNumber parameter.`);
      return;
    }
    if (this.isInPresentationMode || !destArray) {
      this._setCurrentPageNumber(pageNumber, true);
      return;
    }
    let x = 0,
      y = 0;
    let width = 0,
      height = 0,
      widthScale,
      heightScale;
    const changeOrientation = pageView.rotation % 180 !== 0;
    const pageWidth = (changeOrientation ? pageView.height : pageView.width) / pageView.scale / PixelsPerInch.PDF_TO_CSS_UNITS;
    const pageHeight = (changeOrientation ? pageView.width : pageView.height) / pageView.scale / PixelsPerInch.PDF_TO_CSS_UNITS;
    let scale = 0;
    switch (destArray[1].name) {
      case "XYZ":
        x = destArray[2];
        y = destArray[3];
        scale = destArray[4];
        x = x !== null ? x : 0;
        y = y !== null ? y : pageHeight;
        break;
      case "Fit":
      case "FitB":
        scale = "page-fit";
        break;
      case "FitH":
      case "FitBH":
        y = destArray[2];
        scale = "page-width";
        if (y === null && this._location) {
          x = this._location.left;
          y = this._location.top;
        } else if (typeof y !== "number" || y < 0) {
          y = pageHeight;
        }
        break;
      case "FitV":
      case "FitBV":
        x = destArray[2];
        width = pageWidth;
        height = pageHeight;
        scale = "page-height";
        break;
      case "FitR":
        x = destArray[2];
        y = destArray[3];
        width = destArray[4] - x;
        height = destArray[5] - y;
        let hPadding = (/* inlined export .SCROLLBAR_PADDING */40),
          vPadding = (/* inlined export .VERTICAL_PADDING */5);
        if (this.removePageBorders) {
          hPadding = vPadding = 0;
        }
        widthScale = (this.container.clientWidth - hPadding) / width / PixelsPerInch.PDF_TO_CSS_UNITS;
        heightScale = (this.container.clientHeight - vPadding) / height / PixelsPerInch.PDF_TO_CSS_UNITS;
        scale = Math.min(Math.abs(widthScale), Math.abs(heightScale));
        break;
      default:
        console.error(`scrollPageIntoView: "${destArray[1].name}" is not a valid destination type.`);
        return;
    }
    if (!ignoreDestinationZoom) {
      if (scale && scale !== this._currentScale) {
        this.currentScaleValue = scale;
      } else if (this._currentScale === (/* inlined export .UNKNOWN_SCALE */0)) {
        this.currentScaleValue = (/* inlined export .DEFAULT_SCALE_VALUE */"auto");
      }
    }
    if (scale === "page-fit" && !destArray[4]) {
      this.#scrollIntoView(pageView);
      return;
    }
    const boundingRect = [pageView.viewport.convertToViewportPoint(x, y), pageView.viewport.convertToViewportPoint(x + width, y + height)];
    let left = Math.min(boundingRect[0][0], boundingRect[1][0]);
    let top = Math.min(boundingRect[0][1], boundingRect[1][1]);
    if (center) {
      if (center === "both" || center === "vertical") {
        top -= (this.container.clientHeight - Math.abs(boundingRect[1][1] - boundingRect[0][1])) / 2;
      }
      if (center === "both" || center === "horizontal") {
        left -= (this.container.clientWidth - Math.abs(boundingRect[1][0] - boundingRect[0][0])) / 2;
      }
    } else if (!allowNegativeOffset) {
      left = Math.max(left, 0);
      top = Math.max(top, 0);
    }
    this.#scrollIntoView(pageView, {
      left,
      top
    });
  }
  _updateLocation(firstPage) {
    this.#staleLocation = false;
    const currentScale = this._currentScale;
    const currentScaleValue = this._currentScaleValue;
    const normalizedScaleValue = parseFloat(currentScaleValue) === currentScale ? Math.round(currentScale * 10000) / 100 : currentScaleValue;
    const pageNumber = firstPage.id;
    const currentPageView = this._pages[pageNumber - 1];
    const container = this.container;
    const topLeft = currentPageView.getPagePoint(container.scrollLeft - firstPage.x, container.scrollTop - firstPage.y);
    const [left, top] = topLeft;
    let pdfOpenParams = `#page=${pageNumber}`;
    if (!this.isInPresentationMode) {
      pdfOpenParams += `&zoom=${normalizedScaleValue},` + `${Math.round(left)},${Math.round(top)}`;
    }
    this._location = {
      pageNumber,
      scale: normalizedScaleValue,
      top,
      left,
      rotation: this._pagesRotation,
      pdfOpenParams
    };
  }
  update() {
    const visible = this._getVisiblePages();
    const visiblePages = visible.views,
      numVisiblePages = visiblePages.length;
    if (numVisiblePages === 0) {
      return;
    }
    const newCacheSize = Math.max(DEFAULT_CACHE_SIZE, 2 * numVisiblePages + 1);
    this.#buffer.resize(newCacheSize, visible.ids);
    for (const {
      view,
      visibleArea
    } of visiblePages) {
      view.updateVisibleArea(visibleArea);
    }
    for (const view of this.#buffer) {
      if (!visible.ids.has(view.id)) {
        view.updateVisibleArea(null);
      }
    }
    this.renderingQueue.renderHighestPriority(visible);
    const isSimpleLayout = this._spreadMode === SpreadMode.NONE && (this._scrollMode === ScrollMode.PAGE || this._scrollMode === ScrollMode.VERTICAL);
    const currentPageNumber = this._currentPageNumber;
    let stillFullyVisible = false;
    for (const page of visiblePages) {
      if (page.percent < 100) {
        break;
      }
      if (page.id === currentPageNumber && isSimpleLayout) {
        stillFullyVisible = true;
        break;
      }
    }
    this._setCurrentPageNumber(stillFullyVisible ? this._currentPageNumber : visiblePages[0].id);
    this._updateLocation(visible.first);
    this.eventBus.dispatch("updateviewarea", {
      source: this,
      location: this._location
    });
  }
  #switchToEditAnnotationMode() {
    const visible = this._getVisiblePages();
    const pagesToRefresh = [];
    const {
      ids,
      views
    } = visible;
    for (const page of views) {
      const {
        view
      } = page;
      if (!view.hasEditableAnnotations()) {
        ids.delete(view.id);
        continue;
      }
      pagesToRefresh.push(page);
    }
    if (pagesToRefresh.length === 0) {
      return null;
    }
    this.renderingQueue.renderHighestPriority({
      first: pagesToRefresh[0],
      last: pagesToRefresh.at(-1),
      views: pagesToRefresh,
      ids
    });
    return ids;
  }
  containsElement(element) {
    return this.container.contains(element);
  }
  focus() {
    this.container.focus();
  }
  get _isContainerRtl() {
    return getComputedStyle(this.container).direction === "rtl";
  }
  get isInPresentationMode() {
    return this.presentationModeState === PresentationModeState.FULLSCREEN;
  }
  get isChangingPresentationMode() {
    return this.presentationModeState === PresentationModeState.CHANGING;
  }
  get isHorizontalScrollbarEnabled() {
    return this.isInPresentationMode ? false : this.container.scrollWidth > this.container.clientWidth;
  }
  get isVerticalScrollbarEnabled() {
    return this.isInPresentationMode ? false : this.container.scrollHeight > this.container.clientHeight;
  }
  _getVisiblePages() {
    const views = this._scrollMode === ScrollMode.PAGE ? this.#scrollModePageState.pages : this._pages,
      horizontal = this._scrollMode === ScrollMode.HORIZONTAL,
      rtl = horizontal && this._isContainerRtl;
    return getVisibleElements({
      scrollEl: this.container,
      views,
      sortByVisibility: true,
      horizontal,
      rtl
    });
  }
  cleanup() {
    for (const pageView of this._pages) {
      if (pageView.renderingState !== RenderingStates.FINISHED) {
        pageView.reset();
      }
    }
  }
  _cancelRendering() {
    for (const pageView of this._pages) {
      pageView.cancelRendering();
    }
  }
  async #ensurePdfPageLoaded(pageView) {
    if (pageView.pdfPage) {
      return pageView.pdfPage;
    }
    try {
      const pdfPage = await this.pdfDocument.getPage(pageView.id);
      if (!pageView.pdfPage) {
        pageView.setPdfPage(pdfPage);
      }
      return pdfPage;
    } catch (reason) {
      console.error("Unable to get page for page view", reason);
      return null;
    }
  }
  #getScrollAhead(visible) {
    if (visible.first?.id === 1) {
      return true;
    } else if (visible.last?.id === this.pagesCount) {
      return false;
    }
    switch (this._scrollMode) {
      case ScrollMode.PAGE:
        return this.#scrollModePageState.scrollDown;
      case ScrollMode.HORIZONTAL:
        return this.scroll.right;
    }
    return this.scroll.down;
  }
  forceRendering(currentlyVisiblePages) {
    const visiblePages = currentlyVisiblePages || this._getVisiblePages();
    const scrollAhead = this.#getScrollAhead(visiblePages);
    const preRenderExtra = this._spreadMode !== SpreadMode.NONE && this._scrollMode !== ScrollMode.HORIZONTAL;
    const ignoreDetailViews = this.#scaleTimeoutId !== null || this.#scrollTimeoutId !== null && visiblePages.views.some(page => page.detailView?.renderingCancelled);
    const pageView = this.renderingQueue.getHighestPriority(visiblePages, this._pages, scrollAhead, preRenderExtra, ignoreDetailViews);
    if (pageView) {
      this.#ensurePdfPageLoaded(pageView).then(() => {
        this.renderingQueue.renderView(pageView);
      });
      return true;
    }
    return false;
  }
  get hasEqualPageSizes() {
    const firstPageView = this._pages[0];
    for (let i = 1, ii = this._pages.length; i < ii; ++i) {
      const pageView = this._pages[i];
      if (pageView.width !== firstPageView.width || pageView.height !== firstPageView.height) {
        return false;
      }
    }
    return true;
  }
  getPagesOverview() {
    let initialOrientation;
    return this._pages.map(pageView => {
      const viewport = pageView.pdfPage.getViewport({
        scale: 1
      });
      const orientation = isPortraitOrientation(viewport);
      if (initialOrientation === undefined) {
        initialOrientation = orientation;
      } else if (this.enablePrintAutoRotate && orientation !== initialOrientation) {
        return {
          width: viewport.height,
          height: viewport.width,
          rotation: (viewport.rotation - 90) % 360
        };
      }
      return {
        width: viewport.width,
        height: viewport.height,
        rotation: viewport.rotation
      };
    });
  }
  get optionalContentConfigPromise() {
    if (!this.pdfDocument) {
      return Promise.resolve(null);
    }
    if (!this._optionalContentConfigPromise) {
      console.error("optionalContentConfigPromise: Not initialized yet.");
      return this.pdfDocument.getOptionalContentConfig({
        intent: "display"
      });
    }
    return this._optionalContentConfigPromise;
  }
  set optionalContentConfigPromise(promise) {
    if (!(promise instanceof Promise)) {
      throw new Error(`Invalid optionalContentConfigPromise: ${promise}`);
    }
    if (!this.pdfDocument) {
      return;
    }
    if (!this._optionalContentConfigPromise) {
      return;
    }
    this._optionalContentConfigPromise = promise;
    this.refresh(false, {
      optionalContentConfigPromise: promise
    });
    this.eventBus.dispatch("optionalcontentconfigchanged", {
      source: this,
      promise
    });
  }
  get scrollMode() {
    return this._scrollMode;
  }
  set scrollMode(mode) {
    if (this._scrollMode === mode) {
      return;
    }
    if (!isValidScrollMode(mode)) {
      throw new Error(`Invalid scroll mode: ${mode}`);
    }
    if (this.pagesCount > PagesCountLimit.FORCE_SCROLL_MODE_PAGE) {
      return;
    }
    this._previousScrollMode = this._scrollMode;
    this.clearSelection();
    this._scrollMode = mode;
    this.eventBus.dispatch("scrollmodechanged", {
      source: this,
      mode
    });
    this._updateScrollMode(this._currentPageNumber);
  }
  _updateScrollMode(pageNumber = null) {
    const scrollMode = this._scrollMode,
      viewer = this.viewer;
    viewer.classList.toggle("scrollHorizontal", scrollMode === ScrollMode.HORIZONTAL);
    viewer.classList.toggle("scrollWrapped", scrollMode === ScrollMode.WRAPPED);
    if (!this.pdfDocument || !pageNumber) {
      return;
    }
    if (scrollMode === ScrollMode.PAGE) {
      this.#ensurePageViewVisible();
    } else if (this._previousScrollMode === ScrollMode.PAGE) {
      this._updateSpreadMode();
    }
    if (this._currentScaleValue && isNaN(this._currentScaleValue)) {
      this.#setScale(this._currentScaleValue, {
        noScroll: true
      });
    }
    this._setCurrentPageNumber(pageNumber, true);
    this.update();
  }
  get spreadMode() {
    return this._spreadMode;
  }
  set spreadMode(mode) {
    if (this._spreadMode === mode) {
      return;
    }
    if (!isValidSpreadMode(mode)) {
      throw new Error(`Invalid spread mode: ${mode}`);
    }
    this.clearSelection();
    this._spreadMode = mode;
    this.eventBus.dispatch("spreadmodechanged", {
      source: this,
      mode
    });
    this._updateSpreadMode(this._currentPageNumber);
  }
  _updateSpreadMode(pageNumber = null) {
    if (!this.pdfDocument) {
      return;
    }
    const viewer = this.viewer,
      pages = this._pages;
    if (this._scrollMode === ScrollMode.PAGE) {
      this.#ensurePageViewVisible();
    } else {
      viewer.textContent = "";
      if (this._spreadMode === SpreadMode.NONE) {
        for (const pageView of this._pages) {
          viewer.append(pageView.div);
        }
      } else {
        const parity = this._spreadMode - 1;
        let spread = null;
        for (let i = 0, ii = pages.length; i < ii; ++i) {
          if (spread === null) {
            spread = document.createElement("div");
            spread.className = "spread";
            viewer.append(spread);
          } else if (i % 2 === parity) {
            spread = spread.cloneNode(false);
            viewer.append(spread);
          }
          spread.append(pages[i].div);
        }
      }
    }
    if (!pageNumber) {
      return;
    }
    if (this._currentScaleValue && isNaN(this._currentScaleValue)) {
      this.#setScale(this._currentScaleValue, {
        noScroll: true
      });
    }
    this._setCurrentPageNumber(pageNumber, true);
    this.update();
  }
  #getPageAdvance(currentPageNumber, previous = false) {
    switch (this._scrollMode) {
      case ScrollMode.WRAPPED:
        {
          const {
              views
            } = this._getVisiblePages(),
            pageLayout = new Map();
          for (const {
            id,
            y,
            percent,
            widthPercent
          } of views) {
            if (percent === 0 || widthPercent < 100) {
              continue;
            }
            pageLayout.getOrInsertComputed(y, makeArr).push(id);
          }
          for (const yArray of pageLayout.values()) {
            const currentIndex = yArray.indexOf(currentPageNumber);
            if (currentIndex === -1) {
              continue;
            }
            const numPages = yArray.length;
            if (numPages === 1) {
              break;
            }
            if (previous) {
              for (let i = currentIndex - 1, ii = 0; i >= ii; i--) {
                const currentId = yArray[i],
                  expectedId = yArray[i + 1] - 1;
                if (currentId < expectedId) {
                  return currentPageNumber - expectedId;
                }
              }
            } else {
              for (let i = currentIndex + 1, ii = numPages; i < ii; i++) {
                const currentId = yArray[i],
                  expectedId = yArray[i - 1] + 1;
                if (currentId > expectedId) {
                  return expectedId - currentPageNumber;
                }
              }
            }
            if (previous) {
              const firstId = yArray[0];
              if (firstId < currentPageNumber) {
                return currentPageNumber - firstId + 1;
              }
            } else {
              const lastId = yArray[numPages - 1];
              if (lastId > currentPageNumber) {
                return lastId - currentPageNumber + 1;
              }
            }
            break;
          }
          break;
        }
      case ScrollMode.HORIZONTAL:
        {
          break;
        }
      case ScrollMode.PAGE:
      case ScrollMode.VERTICAL:
        {
          if (this._spreadMode === SpreadMode.NONE) {
            break;
          }
          const parity = this._spreadMode - 1;
          if (previous && currentPageNumber % 2 !== parity) {
            break;
          } else if (!previous && currentPageNumber % 2 === parity) {
            break;
          }
          const {
              views
            } = this._getVisiblePages(),
            expectedId = previous ? currentPageNumber - 1 : currentPageNumber + 1;
          for (const {
            id,
            percent,
            widthPercent
          } of views) {
            if (id !== expectedId) {
              continue;
            }
            if (percent > 0 && widthPercent === 100) {
              return 2;
            }
            break;
          }
          break;
        }
    }
    return 1;
  }
  nextPage() {
    const currentPageNumber = this._currentPageNumber,
      pagesCount = this.pagesCount;
    if (currentPageNumber >= pagesCount) {
      return false;
    }
    const advance = this.#getPageAdvance(currentPageNumber, false) || 1;
    this.currentPageNumber = Math.min(currentPageNumber + advance, pagesCount);
    return true;
  }
  previousPage() {
    const currentPageNumber = this._currentPageNumber;
    if (currentPageNumber <= 1) {
      return false;
    }
    const advance = this.#getPageAdvance(currentPageNumber, true) || 1;
    this.currentPageNumber = Math.max(currentPageNumber - advance, 1);
    return true;
  }
  updateScale({
    drawingDelay,
    scaleFactor = null,
    steps = null,
    origin,
    pan = null
  }) {
    if (steps === null && scaleFactor === null) {
      throw new Error("Invalid updateScale options: either `steps` or `scaleFactor` must be provided.");
    }
    if (!this.pdfDocument) {
      return;
    }
    let newScale = this._currentScale;
    if (scaleFactor > 0 && scaleFactor !== 1) {
      newScale = Math.round(newScale * scaleFactor * 100) / 100;
    } else if (steps) {
      const delta = steps > 0 ? (/* inlined export .DEFAULT_SCALE_DELTA */1.1) : 1 / (/* inlined export .DEFAULT_SCALE_DELTA */1.1);
      const round = steps > 0 ? Math.ceil : Math.floor;
      steps = Math.abs(steps);
      do {
        newScale = round((newScale * delta).toFixed(2) * 10) / 10;
      } while (--steps > 0);
    }
    newScale = MathClamp(newScale, (/* inlined export .MIN_SCALE */0.1), (/* inlined export .MAX_SCALE */25));
    this.#setScale(newScale, {
      noScroll: false,
      drawingDelay,
      origin,
      pan
    });
  }
  increaseScale(options = {}) {
    this.updateScale({
      ...options,
      steps: options.steps ?? 1
    });
  }
  decreaseScale(options = {}) {
    this.updateScale({
      ...options,
      steps: -(options.steps ?? 1)
    });
  }
  #updateContainerHeightCss(height = this.container.clientHeight) {
    if (height !== this.#previousContainerHeight) {
      this.#previousContainerHeight = height;
      docStyle.setProperty("--viewer-container-height", `${height}px`);
    }
  }
  #resizeObserverCallback(entries) {
    for (const entry of entries) {
      if (entry.target === this.container) {
        this.#updateContainerHeightCss(Math.floor(entry.borderBoxSize[0].blockSize));
        this.#containerTopLeft = null;
        break;
      }
    }
  }
  get containerTopLeft() {
    return this.#containerTopLeft ||= [this.container.offsetTop, this.container.offsetLeft];
  }
  #cleanupTimeouts() {
    if (this.#scaleTimeoutId !== null) {
      clearTimeout(this.#scaleTimeoutId);
      this.#scaleTimeoutId = null;
    }
    if (this.#scrollTimeoutId !== null) {
      clearTimeout(this.#scrollTimeoutId);
      this.#scrollTimeoutId = null;
    }
  }
  #cleanupSwitchAnnotationEditorMode() {
    this.#switchAnnotationEditorModeAC?.abort();
    this.#switchAnnotationEditorModeAC = null;
    if (this.#switchAnnotationEditorModeTimeoutId !== null) {
      clearTimeout(this.#switchAnnotationEditorModeTimeoutId);
      this.#switchAnnotationEditorModeTimeoutId = null;
    }
  }
  #preloadEditingData(mode) {
    switch (mode) {
      case AnnotationEditorType.STAMP:
        this.#mlManager?.loadModel("altText");
        break;
      case AnnotationEditorType.SIGNATURE:
        this.#signatureManager?.loadSignatures();
        break;
    }
  }
  get annotationEditorMode() {
    return this.#annotationEditorUIManager ? this.#annotationEditorMode : AnnotationEditorType.DISABLE;
  }
  set annotationEditorMode({
    mode,
    editId = null,
    isFromKeyboard = false,
    mustEnterInEditMode = false,
    editComment = false
  }) {
    if (!this.#annotationEditorUIManager) {
      throw new Error(`The AnnotationEditor is not enabled.`);
    }
    if (this.#annotationEditorMode === mode) {
      return;
    }
    if (!isValidAnnotationEditorMode(mode)) {
      throw new Error(`Invalid AnnotationEditor mode: ${mode}`);
    }
    if (!this.pdfDocument) {
      return;
    }
    this.#preloadEditingData(mode);
    const {
      eventBus,
      pdfDocument
    } = this;
    const updater = async () => {
      this.#cleanupSwitchAnnotationEditorMode();
      this.#annotationEditorMode = mode;
      await this.#annotationEditorUIManager.updateMode(mode, editId, true, isFromKeyboard, mustEnterInEditMode, editComment);
      if (mode !== this.#annotationEditorMode || pdfDocument !== this.pdfDocument) {
        return;
      }
      eventBus.dispatch("annotationeditormodechanged", {
        source: this,
        mode
      });
    };
    if (mode === AnnotationEditorType.NONE || this.#annotationEditorMode === AnnotationEditorType.NONE) {
      const isEditing = mode !== AnnotationEditorType.NONE;
      if (!isEditing) {
        this.pdfDocument.annotationStorage.resetModifiedIds();
      }
      this.cleanup();
      for (const pageView of this._pages) {
        pageView.toggleEditingMode(isEditing);
      }
      const idsToRefresh = this.#switchToEditAnnotationMode();
      if (isEditing && idsToRefresh) {
        this.#cleanupSwitchAnnotationEditorMode();
        this.#switchAnnotationEditorModeAC = new AbortController();
        const signal = AbortSignal.any([this.#eventAC.signal, this.#switchAnnotationEditorModeAC.signal]);
        eventBus.on("pagerendered", ({
          pageNumber
        }) => {
          idsToRefresh.delete(pageNumber);
          if (idsToRefresh.size === 0) {
            this.#switchAnnotationEditorModeTimeoutId = setTimeout(updater, 0);
          }
        }, {
          signal,
          ...internalOpt
        });
        return;
      }
    }
    updater();
  }
  refresh(noUpdate = false, updateArgs = Object.create(null)) {
    if (!this.pdfDocument) {
      return;
    }
    for (const pageView of this._pages) {
      pageView.update(updateArgs);
    }
    this.#cleanupTimeouts();
    if (!noUpdate) {
      this.update();
    }
  }
}

;// ./web/pdf_single_page_viewer.js


class PDFSinglePageViewer extends PDFViewer {
  _resetView() {
    super._resetView();
    this._scrollMode = ScrollMode.PAGE;
    this._spreadMode = SpreadMode.NONE;
  }
  set scrollMode(mode) {}
  _updateScrollMode() {}
  set spreadMode(mode) {}
  _updateSpreadMode() {}
}

;// ./web/pdf_viewer.component.js
















globalThis.pdfjsViewer = {
  AnnotationLayerBuilder: AnnotationLayerBuilder,
  DownloadManager: DownloadManager,
  EventBus: EventBus,
  FindState: FindState,
  GenericL10n: genericl10n_GenericL10n,
  LinkTarget: LinkTarget,
  parseQueryString: parseQueryString,
  PDFFindController: PDFFindController,
  PDFHistory: PDFHistory,
  PDFLinkService: PDFLinkService,
  PDFPageView: PDFPageView,
  PDFScriptingManager: PDFScriptingManagerComponents,
  PDFSinglePageViewer: PDFSinglePageViewer,
  PDFViewer: PDFViewer,
  ProgressBar: ProgressBar,
  RenderingStates: RenderingStates,
  ScrollMode: ScrollMode,
  SimpleLinkService: SimpleLinkService,
  SpreadMode: SpreadMode,
  StructTreeLayerBuilder: StructTreeLayerBuilder,
  TextLayerBuilder: TextLayerBuilder,
  XfaLayerBuilder: XfaLayerBuilder
};

export { AnnotationLayerBuilder, DownloadManager, EventBus, FindState, genericl10n_GenericL10n as GenericL10n, LinkTarget, PDFFindController, PDFHistory, PDFLinkService, PDFPageView, PDFScriptingManagerComponents as PDFScriptingManager, PDFSinglePageViewer, PDFViewer, ProgressBar, RenderingStates, ScrollMode, SimpleLinkService, SpreadMode, StructTreeLayerBuilder, TextLayerBuilder, XfaLayerBuilder, parseQueryString };

//# sourceMappingURL=pdf_viewer.mjs.map