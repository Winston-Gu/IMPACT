// Copyright (c) 2025
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* This header must be included by all public headers to ensure correct symbol
 * visibility when building or consuming the shared library.
 */
#ifndef PREDICTIVE_CONTROLLER__VISIBILITY_CONTROL_H_
#define PREDICTIVE_CONTROLLER__VISIBILITY_CONTROL_H_

#if defined _WIN32 || defined __CYGWIN__
#ifdef __GNUC__
#define PREDICTIVE_CONTROLLER_EXPORT __attribute__((dllexport))
#define PREDICTIVE_CONTROLLER_IMPORT __attribute__((dllimport))
#else
#define PREDICTIVE_CONTROLLER_EXPORT __declspec(dllexport)
#define PREDICTIVE_CONTROLLER_IMPORT __declspec(dllimport)
#endif
#ifdef PREDICTIVE_CONTROLLER_BUILDING_DLL
#define PREDICTIVE_CONTROLLER_PUBLIC PREDICTIVE_CONTROLLER_EXPORT
#else
#define PREDICTIVE_CONTROLLER_PUBLIC PREDICTIVE_CONTROLLER_IMPORT
#endif
#define PREDICTIVE_CONTROLLER_PUBLIC_TYPE PREDICTIVE_CONTROLLER_PUBLIC
#define PREDICTIVE_CONTROLLER_LOCAL
#else
#define PREDICTIVE_CONTROLLER_EXPORT __attribute__((visibility("default")))
#define PREDICTIVE_CONTROLLER_IMPORT
#if __GNUC__ >= 4
#define PREDICTIVE_CONTROLLER_PUBLIC __attribute__((visibility("default")))
#define PREDICTIVE_CONTROLLER_LOCAL __attribute__((visibility("hidden")))
#else
#define PREDICTIVE_CONTROLLER_PUBLIC
#define PREDICTIVE_CONTROLLER_LOCAL
#endif
#define PREDICTIVE_CONTROLLER_PUBLIC_TYPE
#endif

#endif // PREDICTIVE_CONTROLLER__VISIBILITY_CONTROL_H_
