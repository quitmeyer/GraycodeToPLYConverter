/* 
 * Author:  Johanan Round
 * Package: Editor Window Fullscreen
 * License: Unity Asset Store EULA (Editor extension asset. Requires 1 license per machine.)
 */

using UnityEngine;
using System;
using System.Diagnostics;
using System.Collections.Generic;

namespace EditorWindowFullscreen
{
    /// <summary>
    /// Contains methods for debugging Editor Window Fullscreen
    /// </summary>
    public sealed class EWFDebugging
    {
        private static EditorFullscreenSettings settings
        {
            get { return EditorFullscreenSettings.settings; }
        }
        public static bool Enabled
        {
            get { return settings.debugModeEnabled; }
        }

        public static string debugLog;
        public static int numWarnings;
        public static int numErrors;
        private static bool debugLogBegun = false;

        private static DateTime logBeginTime;
        private static DateTime logEndTime;
        private static Dictionary<string, DateTime> timerStartTimes;
        public static bool enableTimerLogging = true;
        public static string OSVersion
        {
            get
            {
                string OSName = "";
                string OSVer = "";
#if UNITY_EDITOR_WIN
                    OSName = "Windows (Unknown product name)";
                    string productName = "";
                    string releaseID = "";
                    string buildNum = "";
                    try
                    {
                        var windowsVerRegFolder = @"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows NT\CurrentVersion";
                        productName = Microsoft.Win32.Registry.GetValue(windowsVerRegFolder, "ProductName", "").ToString();
                        releaseID = Microsoft.Win32.Registry.GetValue(windowsVerRegFolder, "ReleaseId", "").ToString();
                        buildNum = Microsoft.Win32.Registry.GetValue(windowsVerRegFolder, "CurrentBuild", "").ToString();
                    }
                    catch { }
                    if (productName != "") OSName = productName;
                    if (releaseID != "") OSVer = releaseID + " ";
                    if (buildNum != "") OSVer += "build " + buildNum;
                    OSVer += " (" + Environment.OSVersion.VersionString + ")";
#elif UNITY_EDITOR_OSX
                OSName = "macOS";
                OSVer = Environment.OSVersion.VersionString;
#elif UNITY_EDITOR_LINUX
                OSName = "Linux";
                OSVer = Environment.OSVersion.VersionString;
#endif
                return OSName + ", " + IntPtr.Size * 8 + "-bit, Version: " + OSVer;
            }
        }
        public static void StartTimer(string description)
        {
            if (!enableTimerLogging) return;
            if (timerStartTimes == null) timerStartTimes = new Dictionary<string, DateTime>();
            if (timerStartTimes.ContainsKey(description))
                timerStartTimes[description] = DateTime.Now;
            else
                timerStartTimes.Add(description, DateTime.Now);
        }

        public static void LogTime(string description)
        {
            LogTime(description, true, new StackFrame(1, true));
        }

        public static void LogTime(string description, bool restartTimer)
        {
            LogTime(description, restartTimer, new StackFrame(1, true));
        }

        private static void LogTime(string description, bool restartTimer, StackFrame stack)
        {
            if (!enableTimerLogging) return;
            string timerDuration;
            if (timerStartTimes == null || !timerStartTimes.ContainsKey(description)) timerDuration = "TIMER NOT STARTED ";
            else
            {
                timerDuration = (DateTime.Now - timerStartTimes[description]).Milliseconds.ToString();
                if (restartTimer) timerStartTimes[description] = DateTime.Now;
                else timerStartTimes.Remove(description);
            }
            LogLine("Timer: " + timerDuration + "ms, \"" + description + "\"", stack);
        }

        public static void Begin()
        {
            Begin("", new StackFrame(1, true));
        }
        public static void Begin(string existingLog)
        {
            Begin(existingLog, new StackFrame(1, true));
        }
        private static void Begin(string existingLog, StackFrame stack)
        {
#if UNITY_EDITOR_OSX
            SystemDisplay.EnableDebugging(Enabled);
#endif

            debugLog = "";
            numWarnings = 0;
            numErrors = 0;
            logBeginTime = DateTime.Now;

            if (timerStartTimes != null) timerStartTimes.Clear();
            else timerStartTimes = new Dictionary<string, DateTime>();

            if (settings.debugModeEnabled && debugLogBegun == false)
            {
                debugLogBegun = true;
                if (stack == null) stack = new StackFrame(1, true);
                if (String.IsNullOrEmpty(existingLog))
                {
                    //Start debugging and add system information to the log
                    LogLine("Debugging Enabled - Editor Window Fullscreen - To disable debugging, uncheck it in the 'Fullscreen Window Settings' window.", stack);
                    Log("OS: " + OSVersion + ".");
                    Log(" Unity Version: " + Application.unityVersion + ".");
                    Log(" EWF Version: " + EditorFullscreenSettings.Version + "\n");
                    Log("-------------------------------------\n");
                }
                else
                {
                    //Start debugging with existing log
                    LogLine("\nDebugging Resuming - Editor Window Fullscreen - Existing Log");
                }
            }
        }
        public static void Log(string message)
        {
            if (!debugLogBegun) return;
            debugLog += message;
        }
        public static void LogLine(string message)
        {
            LogLine(message, new StackFrame(1, true));
        }
        public static void LogLine(string message, StackFrame stack)
        {
            if (!debugLogBegun) return;
            if (stack == null) stack = new StackFrame(1, true);
            debugLog += message + " " + StackCallString(stack) + "\n";
        }
        public static void LogLine(string message, int skipExtraStackFrames, int logStackFrames)
        {
            if (!debugLogBegun) return;
            string stackCallString = "";
            for (int i = 1; i <= logStackFrames; i++)
            {
                var stack = new StackFrame(skipExtraStackFrames + i, true);
                stackCallString += StackCallString(stack);
                if (i < logStackFrames) stackCallString += " << ";
            }
            debugLog += message + " " + stackCallString + "\n";
        }
        public static void LogWarning(string message)
        {
            LogWarning(message, 1);
        }
        public static void LogWarning(string message, bool printWarningToConsole)
        {
            if (!Enabled) return;
            if (printWarningToConsole) UnityEngine.Debug.LogWarning(message);
            LogWarning(message, 1);

        }
        private static void LogWarning(string message, int skipExtraFrames)
        {
            if (!Enabled) return;
            var stack = new StackFrame(1 + skipExtraFrames, true);
            LogLine("--------------\nWarning: " + message + "\n--------------", stack);
            numWarnings++;
        }
        public static void LogError(string message)
        {
            LogError(message, 0);
        }
        public static void LogError(string message, bool printErrorToConsole)
        {
            if (!Enabled) return;
            LogError(message, 0);
            if (printErrorToConsole) UnityEngine.Debug.LogError(message);
        }
        public static void LogError(string message, Exception e)
        {
            if (!Enabled) return;
            LogError(message, 0);
            UnityEngine.Debug.LogError(message + "\n" + e.ToString());
        }
        public static void LogError(string message, int skipExtraStackFrames)
        {
            var stack = new StackFrame(skipExtraStackFrames + 1, true);
            LogLine("--------------\nError: " + message + "\n--------------", stack);
            numErrors++;
        }
        public static void PrintLog()
        {
            if (settings.debugModeEnabled && debugLogBegun)
            {
                logEndTime = DateTime.Now;
                debugLog += "\n----- End of Debug Log -----";
                debugLog += " (Time since start: " + (logEndTime - logBeginTime).TotalMilliseconds + "ms)";
                var debugLogLines = debugLog.Split("\n".ToCharArray());
                string hundredLines = "";
                int pageNum = 1;
                for (int i = 0; i < debugLogLines.Length; i++)
                {
                    hundredLines += debugLogLines[i] + "\n";
                    if ((i + 1) % 80 == 0 || i == debugLogLines.Length - 1)
                    {
                        UnityEngine.Debug.Log(hundredLines);
                        pageNum++;
                        hundredLines = "...debug log continued (page " + pageNum + ")...\n";
                    }
                }
            }
            debugLogBegun = false;
        }
        private static string StackCallString(StackFrame stack)
        {
            return "(Line " + stack.GetFileLineNumber() + ", " + stack.GetMethod().ReflectedType.Name + ", " + stack.GetMethod() + ")";
        }
    }
}