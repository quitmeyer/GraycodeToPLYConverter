/* 
 * Author:  Johanan Round
 */

using UnityEngine;
using UnityEditor;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using EditorWindowFullscreen;

public partial class SystemDisplay
{
    /// <summary>
    /// Windows-specific methods of SystemDisplay
    /// </summary>
    protected class WindowsDisplay
    {
#if UNITY_EDITOR_WIN
        private static void LogWin32Error()
        {
            LogWin32Error("");
        }
        private static void LogWin32Error(string message)
        {
            int err = Marshal.GetLastWin32Error();
            if (err != 0)
            {
                var errDesc = "";
                if (err == 1400) errDesc = "Invalid window handle";
                var errMsg = "Win32 Error: " + err + " '" + errDesc + "' - " + message;
                if (EWFDebugging.Enabled) Debug.LogError(errMsg);
                EWFDebugging.LogError(errMsg, 1);
            }
        }

        public static List<SystemDisplay> GetAllDisplays()
        {
            List<SystemDisplay> allDisplays = new List<SystemDisplay>();
            IntPtr hMainWindowMonitor = IntPtr.Zero;
            IntPtr mainWindowHandle = IntPtr.Zero;
            EWFDebugging.LogLine("Getting all displays.", 0, 4);
            try
            {
                mainWindowHandle = GetProcessMainWindow();
                if (mainWindowHandle != IntPtr.Zero)
                {
                    var mainWindowMonitorInfoEx = MonitorInfoEx.CreateWithDefaults();
                    hMainWindowMonitor = MonitorFromWindow(mainWindowHandle, MONITOR_DEFAULTTONEAREST);
                    LogWin32Error("Error finding main window monitor");
                    if (hMainWindowMonitor != IntPtr.Zero)
                    {
                        GetMonitorInfo(hMainWindowMonitor, ref mainWindowMonitorInfoEx);
                        LogWin32Error("Error getting main window monitor info");
                    }
                }
                else
                {
                    EWFDebugging.LogError("Could not find the process main window handle.");
                }
            }
            catch (Exception e)
            {
                string err = "Error finding the main window monitor. " + e.ToString();
                Debug.LogError(err);
                EWFDebugging.LogError(err);
            }

            var deviceDisplayMonitorCount = new Dictionary<string, uint>();
            EnumDisplayMonitors(IntPtr.Zero, IntPtr.Zero,
                delegate (IntPtr hMonitor, IntPtr hdcMonitor, ref RectStruct lprcMonitor, IntPtr dwData)
                {
                    try
                    {
                        //Get the monitor info
                        var monitorInfoEx = MonitorInfoEx.CreateWithDefaults();
                        GetMonitorInfo(hMonitor, ref monitorInfoEx);
                        LogWin32Error();

                        //Get the associated display device
                        bool mirroringDriver = false;
                        bool attachedToDesktop = false;
                        string deviceName = monitorInfoEx.DeviceName;

                        if (!deviceDisplayMonitorCount.ContainsKey(deviceName)) deviceDisplayMonitorCount.Add(deviceName, 0);
                        deviceDisplayMonitorCount[deviceName] += 1;

                        var displayDevice = Display_Device.CreateWithDefaults();
                        int displayMonitor = 0;
                        for (uint id = 0; EnumDisplayDevices(deviceName, id, ref displayDevice, 0); id++)
                        {
                            attachedToDesktop = ((displayDevice.StateFlags & DisplayDeviceStateFlags.AttachedToDesktop) == DisplayDeviceStateFlags.AttachedToDesktop);

                            if (attachedToDesktop)
                            {
                                displayMonitor++;
                                if (displayMonitor == deviceDisplayMonitorCount[deviceName])
                                {
                                    mirroringDriver = ((displayDevice.StateFlags & DisplayDeviceStateFlags.MirroringDriver) == DisplayDeviceStateFlags.MirroringDriver);
                                    break; //Found the display device which matches the monitor
                                }
                            }

                            displayDevice.Size = Marshal.SizeOf(displayDevice);
                        }

                        //Skip the monitor if it's a pseudo monitor
                        if (mirroringDriver) return true;

                        //Store the monitor info in a SystemDisplay object
                        var display = new SystemDisplay();
                        display.Name = displayDevice.DeviceString;
                        display.AttachedToDesktop = attachedToDesktop; //Should always be true within EnumDisplayMonitors
                        display.IsPrimary = monitorInfoEx.Flags == (uint)1;
                        display.HasMainWindow = (hMonitor == hMainWindowMonitor);
                        display.Bounds = RectFromRectStruct(lprcMonitor);
                        display.WorkArea = RectFromRectStruct(monitorInfoEx.WorkAreaBounds);

                        var devMode = new DEVMODE();
                        EnumDisplaySettings(monitorInfoEx.DeviceName, ENUM_CURRENT_SETTINGS, ref devMode);
                        display.PixelWidth = devMode.dmPelsWidth;
                        display.PixelHeight = devMode.dmPelsHeight;

                        //Add the SystemDisplay to allDisplays
                        allDisplays.Add(display);

                    }
                    catch (Exception e)
                    {
                        Debug.LogException(e);
                        EWFDebugging.LogError(e.ToString());
                    }
                    LogWin32Error();

                    return true; //Continue the enumeration
                }, IntPtr.Zero);
            LogWin32Error();

            //Calculate physical bounds
            foreach (var display in allDisplays)
            {
                Rect physicalBounds = display.Bounds;
                physicalBounds.width = display.PixelWidth;
                physicalBounds.height = display.PixelHeight;
                Vector2 displayTopLeft = new Vector2(display.Bounds.xMin, display.Bounds.yMin);

                var displayTopLeftPhysical = GetPhysicalPoint(mainWindowHandle, displayTopLeft);
                physicalBounds.x = displayTopLeftPhysical.x;
                physicalBounds.y = displayTopLeftPhysical.y;
                display.PhysicalBounds = physicalBounds;
            }

            return allDisplays;
        }

        private static Rect RectFromRectStruct(RectStruct rectStruct)
        {
            return new Rect(rectStruct.Left, rectStruct.Top, rectStruct.Right - rectStruct.Left, rectStruct.Bottom - rectStruct.Top);
        }

        /// Makes sure a window covers the taskbar when it is fullscreen
        internal static bool MakeWindowCoverTaskBar(EditorWindow editorWindow, SystemDisplay display)
        {
            if (editorWindow == null) return false;
            return MakeWindowCoverTaskBar(editorWindow, null, null, display);
        }

        /// Makes sure a window covers the taskbar when it is fullscreen
        internal static bool MakeWindowCoverTaskBar(string windowClass, string windowTitle, SystemDisplay display)
        {
            return MakeWindowCoverTaskBar(null, windowClass, windowTitle, display);
        }

        /// Makes sure a window covers the taskbar when it is fullscreen
        private static bool MakeWindowCoverTaskBar(EditorWindow editorWindow, string windowClass, string windowTitle, SystemDisplay display)
        {
            IntPtr windowHandle = IntPtr.Zero;
            EWFDebugging.StartTimer("Making window cover taskbar");
            EWFDebugging.LogLine("Making window cover taskbar. " + (editorWindow == null ? "WindowTitle: " + (windowTitle == null ? "null" : windowTitle) : "EditorWindow: '" + editorWindow.GetWindowTitle() + "' Window Type: '" + editorWindow.GetType() + "' with class: '" + (windowClass == null ? "null" : windowClass) + "'"));
            if (editorWindow == null)
            {
                string fullscreenWindowClass = windowClass != null ? windowClass : "UnityPopupWndClass";
                windowHandle = GetProcessWindow(fullscreenWindowClass, windowTitle, true);
                if (windowHandle == IntPtr.Zero) windowHandle = GetProcessWindow(null, windowTitle, true);
            }
            else
            {
                if (windowClass == null)
                    windowHandle = GetProcessWindow(editorWindow);
                else
                    windowHandle = GetProcessWindow(windowClass, editorWindow);
            }

            if (windowHandle == IntPtr.Zero)
            {
                EWFDebugging.LogError("Couldn't find window handle.");
                return false;
            }

            IntPtr existingStyle = GetWindowLongPtr(windowHandle, GWL_STYLE);
            IntPtr existingExStyle = GetWindowLongPtr(windowHandle, GWL_EXSTYLE);
            if (editorWindow != null)
            {
                var state = EditorFullscreenState.FindWindowState(editorWindow);
                if (state.OriginalStyle == 0) state.OriginalStyle = (int)existingStyle;
                if (state.OriginalExStyle == 0) state.OriginalExStyle = (int)existingExStyle;
            }

            if (EWFDebugging.Enabled)
            {
                EWFDebugging.LogLine("before Style: " + WindowStyleToString(existingStyle));
                EWFDebugging.LogLine("before ExStyle: " + WindowExStyleToString(existingExStyle));
            }

            SetWindowLongPtr(windowHandle, GWL_STYLE, (IntPtr)(WS_POPUP | WS_VISIBLE | ((uint)existingStyle & (WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_OVERLAPPED))));
            LogWin32Error("Error setting window style");

            SetWindowLongPtr(windowHandle, GWL_EXSTYLE, (IntPtr)((uint)existingExStyle & (WS_EX_LEFT | WS_EX_LTRREADING | WS_EX_RIGHTSCROLLBAR)));
            LogWin32Error("Error setting window ex style");

            SetWindowPos(windowHandle, IntPtr.Zero, (int)display.Bounds.x, (int)display.Bounds.y, (int)display.Bounds.width, (int)display.Bounds.height,
                         SWP.NOZORDER | SWP.FRAMECHANGED | SWP.NOACTIVATE);
            LogWin32Error("Error setting window position");

            if (EWFDebugging.Enabled)
            {
                existingStyle = GetWindowLongPtr(windowHandle, GWL_STYLE);
                existingExStyle = GetWindowLongPtr(windowHandle, GWL_EXSTYLE);
                EWFDebugging.LogLine("after Style: " + WindowStyleToString(existingStyle));
                EWFDebugging.LogLine("after ExStyle: " + WindowExStyleToString(existingExStyle));
                EWFDebugging.LogTime("Making window cover taskbar");
            }

            return true;
        }
        internal static void RestoreWindowStyle(EditorWindow editorWindow, Rect origPosition)
        {
                if (editorWindow == null) return;
                var windowHandle = GetProcessWindow(editorWindow);
                var state = EditorFullscreenState.FindWindowState(editorWindow);

                if (state != null && windowHandle != IntPtr.Zero)
                {
                    SetWindowLongPtr(windowHandle, GWL_STYLE, (IntPtr)state.OriginalStyle);
                    LogWin32Error("Error setting window style");

                    SetWindowPos(windowHandle, IntPtr.Zero, 0, 0, 0, 0, SWP.NOZORDER | SWP.FRAMECHANGED | SWP.NOACTIVATE | SWP.NOMOVE | SWP.NOSIZE);
                }
        }

        private static IntPtr hMenu = IntPtr.Zero;

        internal static void SetMainWindowFullscreenStyle(EditorFullscreenState.WindowFullscreenState fullscreenState, bool showMenuBar)
        {
            var windowHandle = GetProcessMainWindow();
            
            if (fullscreenState != null && windowHandle != IntPtr.Zero)
            {
                var menuHandle = GetMenu(windowHandle);
                if (showMenuBar)
                {
                    if (menuHandle == IntPtr.Zero && hMenu != IntPtr.Zero) SetMenu(windowHandle, hMenu);
                }
                else
                {
                    if (menuHandle != IntPtr.Zero)
                    {
                        hMenu = menuHandle;
                        SetMenu(windowHandle, IntPtr.Zero);
                    }
                }
                var setStyle = showMenuBar ? ((uint)fullscreenState.OriginalStyle & (~WS_CAPTION)) | WS_SYSMENU : (uint)fullscreenState.OriginalStyle & (~(WS_SYSMENU | WS_CAPTION | WS_BORDER | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX));
                SetWindowLongPtr(windowHandle, GWL_STYLE, (IntPtr)(setStyle)); //Always make resizable when reloading style.
                LogWin32Error("Error setting main window style");
                SetWindowLongPtr(windowHandle, GWL_EXSTYLE, (IntPtr)fullscreenState.OriginalExStyle);
                LogWin32Error("Error setting main window ex style");
                SetWindowPos(windowHandle, IntPtr.Zero, 0, 0, 0, 0, SWP.NOZORDER | SWP.FRAMECHANGED | SWP.NOACTIVATE | SWP.NOMOVE | SWP.NOSIZE);
                
                if (EWFDebugging.Enabled)
                {
                    EWFDebugging.LogLine("Loaded Main Window Style: " + WindowStyleToString((IntPtr)fullscreenState.OriginalStyle));
                    EWFDebugging.LogLine("Loaded Main Window ExStyle: " + WindowExStyleToString((IntPtr)fullscreenState.OriginalExStyle));
                }
            }
        }

        internal static void SaveMainWindowStyleInState(EditorFullscreenState.WindowFullscreenState fullscreenState)
        {
            var windowHandle = GetProcessMainWindow();
            if (fullscreenState != null && windowHandle != IntPtr.Zero)
            {
                var existingStyle = GetWindowLongPtr(windowHandle, GWL_STYLE);
                LogWin32Error("Error getting main window style");
                var existingExStyle = GetWindowLongPtr(windowHandle, GWL_EXSTYLE);
                LogWin32Error("Error getting main window ex style");

                fullscreenState.OriginalStyle = (int)existingStyle;
                fullscreenState.OriginalExStyle = (int)existingExStyle;

                if (EWFDebugging.Enabled)
                {
                    EWFDebugging.LogLine("Saved Main Window Style: " + WindowStyleToString(existingStyle));
                    EWFDebugging.LogLine("Saved Main Window ExStyle: " + WindowExStyleToString(existingExStyle));
                }
            }
        }

        internal static void LoadMainWindowStyleInState(EditorFullscreenState.WindowFullscreenState fullscreenState, bool makeResizable)
        {
            var windowHandle = GetProcessMainWindow();
            if (fullscreenState != null && windowHandle != IntPtr.Zero)
            {
                var setStyle = makeResizable ? (uint)fullscreenState.OriginalStyle | WS_SIZEBOX | WS_CAPTION | WS_BORDER | WS_OVERLAPPED : (uint)fullscreenState.OriginalStyle;
                SetWindowLongPtr(windowHandle, GWL_STYLE, (IntPtr)(setStyle)); //Always make resizable when reloading style.
                LogWin32Error("Error setting main window style");
                SetWindowLongPtr(windowHandle, GWL_EXSTYLE, (IntPtr)fullscreenState.OriginalExStyle);
                LogWin32Error("Error setting main window ex style");
                SetWindowPos(windowHandle, IntPtr.Zero, 0, 0, 0, 0, SWP.NOZORDER | SWP.FRAMECHANGED | SWP.NOACTIVATE | SWP.NOMOVE | SWP.NOSIZE);

                if (EWFDebugging.Enabled)
                {
                    EWFDebugging.LogLine("Loaded Main Window Style: " + WindowStyleToString((IntPtr)fullscreenState.OriginalStyle));
                    EWFDebugging.LogLine("Loaded Main Window ExStyle: " + WindowExStyleToString((IntPtr)fullscreenState.OriginalExStyle));
                }
            }
        }

        private static string WindowStyleToString(IntPtr windowStyle)
        {
            string styles = "";
            var winStyle = (uint)windowStyle;
            styles += ((winStyle & WS_BORDER) == WS_BORDER) ? "WS_BORDER" + " " : "";
            styles += ((winStyle & WS_CAPTION) == WS_CAPTION) ? "WS_CAPTION" + " " : "";
            styles += ((winStyle & WS_CHILD) == WS_CHILD) ? "WS_CHILD" + " " : "";
            styles += ((winStyle & WS_CLIPCHILDREN) == WS_CLIPCHILDREN) ? "WS_CLIPCHILDREN" + " " : "";
            styles += ((winStyle & WS_CLIPSIBLINGS) == WS_CLIPSIBLINGS) ? "WS_CLIPSIBLINGS" + " " : "";
            styles += ((winStyle & WS_DISABLED) == WS_DISABLED) ? "WS_DISABLED" + " " : "";
            styles += ((winStyle & WS_DLGFRAME) == WS_DLGFRAME) ? "WS_DLGFRAME" + " " : "";
            styles += ((winStyle & WS_GROUP) == WS_GROUP) ? "WS_GROUP" + " " : "";
            styles += ((winStyle & WS_HSCROLL) == WS_HSCROLL) ? "WS_HSCROLL" + " " : "";
            styles += ((winStyle & WS_ICONIC) == WS_ICONIC) ? "WS_ICONIC" + " " : "";
            styles += ((winStyle & WS_MAXIMIZE) == WS_MAXIMIZE) ? "WS_MAXIMIZE" + " " : "";
            styles += ((winStyle & WS_MAXIMIZEBOX) == WS_MAXIMIZEBOX) ? "WS_MAXIMIZEBOX" + " " : "";
            styles += ((winStyle & WS_MINIMIZE) == WS_MINIMIZE) ? "WS_MINIMIZE" + " " : "";
            styles += ((winStyle & WS_MINIMIZEBOX) == WS_MINIMIZEBOX) ? "WS_MINIMIZEBOX" + " " : "";
            styles += ((winStyle & WS_OVERLAPPED) == WS_OVERLAPPED) ? "WS_OVERLAPPED" + " " : "";
            styles += ((winStyle & WS_POPUP) == WS_POPUP) ? "WS_POPUP" + " " : "";
            styles += ((winStyle & WS_SIZEBOX) == WS_SIZEBOX) ? "WS_SIZEBOX" + " " : "";
            styles += ((winStyle & WS_SYSMENU) == WS_SYSMENU) ? "WS_SYSMENU" + " " : "";
            styles += ((winStyle & WS_TABSTOP) == WS_TABSTOP) ? "WS_TABSTOP" + " " : "";
            styles += ((winStyle & WS_THICKFRAME) == WS_THICKFRAME) ? "WS_THICKFRAME" + " " : "";
            styles += ((winStyle & WS_TILED) == WS_TILED) ? "WS_TILED" + " " : "";
            styles += ((winStyle & WS_VISIBLE) == WS_VISIBLE) ? "WS_VISIBLE" + " " : "";
            styles += ((winStyle & WS_VSCROLL) == WS_VSCROLL) ? "WS_VSCROLL" + " " : "";
            return styles;
        }

        private static string WindowExStyleToString(IntPtr windowExStyle)
        {
            string styles = "";
            var winStyle = (uint)windowExStyle;
            styles += ((winStyle & WS_EX_ACCEPTFILES) == WS_EX_ACCEPTFILES) ? "WS_EX_ACCEPTFILES" + " " : "";
            styles += ((winStyle & WS_EX_APPWINDOW) == WS_EX_APPWINDOW) ? "WS_EX_APPWINDOW" + " " : "";
            styles += ((winStyle & WS_EX_CLIENTEDGE) == WS_EX_CLIENTEDGE) ? "WS_EX_CLIENTEDGE" + " " : "";
            styles += ((winStyle & WS_EX_COMPOSITED) == WS_EX_COMPOSITED) ? "WS_EX_COMPOSITED" + " " : "";
            styles += ((winStyle & WS_EX_CONTEXTHELP) == WS_EX_CONTEXTHELP) ? "WS_EX_CONTEXTHELP" + " " : "";
            styles += ((winStyle & WS_EX_CONTROLPARENT) == WS_EX_CONTROLPARENT) ? "WS_EX_CONTROLPARENT" + " " : "";
            styles += ((winStyle & WS_EX_DLGMODALFRAME) == WS_EX_DLGMODALFRAME) ? "WS_EX_DLGMODALFRAME" + " " : "";
            styles += ((winStyle & WS_EX_LAYERED) == WS_EX_LAYERED) ? "WS_EX_LAYERED" + " " : "";
            styles += ((winStyle & WS_EX_LAYOUTRTL) == WS_EX_LAYOUTRTL) ? "WS_EX_LAYOUTRTL" + " " : "";
            styles += ((winStyle & WS_EX_LEFT) == WS_EX_LEFT) ? "WS_EX_LEFT" + " " : "";
            styles += ((winStyle & WS_EX_LEFTSCROLLBAR) == WS_EX_LEFTSCROLLBAR) ? "WS_EX_LEFTSCROLLBAR" + " " : "";
            styles += ((winStyle & WS_EX_LTRREADING) == WS_EX_LTRREADING) ? "WS_EX_LTRREADING" + " " : "";
            styles += ((winStyle & WS_EX_MDICHILD) == WS_EX_MDICHILD) ? "WS_EX_MDICHILD" + " " : "";
            styles += ((winStyle & WS_EX_NOACTIVATE) == WS_EX_NOACTIVATE) ? "WS_EX_NOACTIVATE" + " " : "";
            styles += ((winStyle & WS_EX_NOINHERITLAYOUT) == WS_EX_NOINHERITLAYOUT) ? "WS_EX_NOINHERITLAYOUT" + " " : "";
            styles += ((winStyle & WS_EX_OVERLAPPEDWINDOW) == WS_EX_OVERLAPPEDWINDOW) ? "WS_EX_OVERLAPPEDWINDOW" + " " : "";
            styles += ((winStyle & WS_EX_PALETTEWINDOW) == WS_EX_PALETTEWINDOW) ? "WS_EX_PALETTEWINDOW" + " " : "";
            styles += ((winStyle & WS_EX_RIGHT) == WS_EX_RIGHT) ? "WS_EX_RIGHT" + " " : "";
            styles += ((winStyle & WS_EX_RIGHTSCROLLBAR) == WS_EX_RIGHTSCROLLBAR) ? "WS_EX_RIGHTSCROLLBAR" + " " : "";
            styles += ((winStyle & WS_EX_RTLREADING) == WS_EX_RTLREADING) ? "WS_EX_RTLREADING" + " " : "";
            styles += ((winStyle & WS_EX_STATICEDGE) == WS_EX_STATICEDGE) ? "WS_EX_STATICEDGE" + " " : "";
            styles += ((winStyle & WS_EX_TOOLWINDOW) == WS_EX_TOOLWINDOW) ? "WS_EX_TOOLWINDOW" + " " : "";
            styles += ((winStyle & WS_EX_TOPMOST) == WS_EX_TOPMOST) ? "WS_EX_TOPMOST" + " " : "";
            styles += ((winStyle & WS_EX_TRANSPARENT) == WS_EX_TRANSPARENT) ? "WS_EX_TRANSPARENT" + " " : "";
            styles += ((winStyle & WS_EX_WINDOWEDGE) == WS_EX_WINDOWEDGE) ? "WS_EX_WINDOWEDGE" + " " : "";
            return styles;
        }

        internal static bool WindowIsFullscreenOnDisplay(EditorWindow editorWindow, string windowTitle, SystemDisplay display)
        {
            EWFDebugging.StartTimer("WindowIsFullscreenOnDisplay");
            EWFDebugging.Log("Checking if window is fullscreen on display. " + (editorWindow == null ? "WindowTitle: " + (windowTitle == null ? "null" : windowTitle) : "EditorWindow: " + editorWindow.GetWindowTitle() + " Identifier: " + editorWindow.GetIdentifierTitle()) + "\n");

            IntPtr windowHandle = GetProcessWindow(null, editorWindow);
            LogWin32Error("Error getting window handle.");
            if (windowHandle == IntPtr.Zero)
            {
                EWFDebugging.Log("Couldn't find window handle. Zero pointer.\n");
                return false;
            }

            Rect winPhysBounds = GetWindowPhysicalBounds(windowHandle);
            Rect displayPhysBounds = display.PhysicalBounds;

            float padding = 1;
            winPhysBounds.xMin -= padding;
            winPhysBounds.xMax += padding;
            winPhysBounds.yMin -= padding;
            winPhysBounds.yMax += padding;

            EWFDebugging.LogTime("WindowIsFullscreenOnDisplay", false);
            return winPhysBounds.Contains(displayPhysBounds);
        }

        internal static Rect GetWindowBounds(IntPtr windowHandle)
        {
            RectStruct winRectStruct = new RectStruct();
            GetWindowRect(windowHandle, ref winRectStruct);
            LogWin32Error("Error getting window rect.");
            Rect winRect = RectFromRectStruct(winRectStruct);
            return winRect;
        }

        internal static Rect GetWindowPhysicalBounds(EditorWindow editorWindow)
        {
            EWFDebugging.Log("Getting editor window physical bounds. " + "EditorWindow: " + editorWindow.GetWindowTitle());
            IntPtr windowHandle = GetProcessWindow(null, editorWindow);
            Rect winPhysBounds = GetWindowPhysicalBounds(windowHandle);
            return winPhysBounds;
        }

        internal static Rect GetWindowPhysicalBounds(IntPtr windowHandle)
        {
            RectStruct winRect = new RectStruct();
            GetWindowRect(windowHandle, ref winRect);
            LogWin32Error("Error getting window rect." + " Top: " + winRect.Top + " Bottom: " + winRect.Bottom);
            POINT winTopLeft = GetPhysicalPoint(windowHandle, new POINT(winRect.Left, winRect.Top));
            POINT win100 = GetPhysicalPoint(windowHandle, new POINT(winRect.Left + 100, winRect.Top));
            float scalingFactor = (win100.X - winTopLeft.X) / 100f;
            Rect winPhysicalBounds = new Rect(winTopLeft.X, winTopLeft.Y, Mathf.CeilToInt((winRect.Right - winRect.Left) * scalingFactor), Mathf.CeilToInt((winRect.Bottom - winRect.Top) * scalingFactor));

            return winPhysicalBounds;
        }

        internal static Vector2 GetPhysicalPoint(Vector2 logicalPoint)
        {
            IntPtr mainWindowHandle = GetProcessMainWindow();
            RectStruct winRect = new RectStruct();
            GetWindowRect(mainWindowHandle, ref winRect);
            POINT winTopLeft = GetPhysicalPoint(mainWindowHandle, new POINT(winRect.Left, winRect.Top));
            POINT win100 = GetPhysicalPoint(mainWindowHandle, new POINT(winRect.Left + 100, winRect.Top));
            float scalingFactor = (win100.X - winTopLeft.X) / 100f;

            Vector2 physicalPoint = logicalPoint * scalingFactor;
            physicalPoint.x = Mathf.RoundToInt(physicalPoint.x);
            physicalPoint.y = Mathf.RoundToInt(physicalPoint.y);
            return physicalPoint;
        }
        internal static Vector2 GetPhysicalPoint(IntPtr windowHandle, Vector2 logicalPoint)
        {
            POINT physPoint = GetPhysicalPoint(windowHandle, new POINT((int)logicalPoint.x, (int)logicalPoint.y));
            return new Vector2(physPoint.X, physPoint.Y);
        }
        internal static POINT GetPhysicalPoint(IntPtr windowHandle, POINT logicalPoint)
        {
            POINT physicalPoint = new POINT(logicalPoint.X, logicalPoint.Y);
            try
            {
                LogicalToPhysicalPointForPerMonitorDPI(windowHandle, ref physicalPoint);
            }
            catch
            {
                try
                {
                    LogicalToPhysicalPoint(windowHandle, ref physicalPoint);
                }
                catch
                {
                    //Point remains logical
                }
            }
            LogWin32Error("Error finding physical point");
            return physicalPoint;
        }

        /**** Windows API Calls ****/

        private const int CCHDEVICENAME = 32; //Size of device name string

        delegate bool MonitorEnumProc(IntPtr hMonitor, IntPtr hdcMonitor, ref RectStruct lprcMonitor, IntPtr dwData);

        [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        static extern bool EnumDisplayMonitors(IntPtr hdc, IntPtr lprcClip, MonitorEnumProc lpfnEnum, IntPtr dwData);

        [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        static extern bool GetMonitorInfo(IntPtr hMonitor, ref MonitorInfoEx lpmi);

        [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        static extern bool EnumDisplayDevices(string lpDevice, uint iDevNum, ref Display_Device lpDisplayDevice, uint dwFlags);

        [DllImport("user32.dll", SetLastError = true)]
        public static extern bool EnumDisplaySettings(string deviceName, int modeNum, ref DEVMODE devMode);
        const int ENUM_CURRENT_SETTINGS = -1;
        const int ENUM_REGISTRY_SETTINGS = -2;

        [DllImport("user32.dll", SetLastError = true)]
        static extern IntPtr GetDesktopWindow();

        [DllImport("user32.dll", SetLastError = true)]
        static extern IntPtr MonitorFromWindow(IntPtr hwnd, uint dwFlags);
        const int MONITOR_DEFAULTTONULL = 0;
        const int MONITOR_DEFAULTTOPRIMARY = 1;
        const int MONITOR_DEFAULTTONEAREST = 2;

        [DllImport("user32.dll", SetLastError = true)]
        public static extern bool GetWindowRect(IntPtr hwnd, ref RectStruct rect);

        [DllImport("user32.dll", EntryPoint = "GetWindowLong", SetLastError = true)]
        private static extern IntPtr GetWindowLongPtr32(IntPtr hWnd, int nIndex);

        [DllImport("user32.dll", EntryPoint = "GetWindowLongPtr", SetLastError = true)]
        private static extern IntPtr GetWindowLongPtr64(IntPtr hWnd, int nIndex);

        public static IntPtr GetWindowLongPtr(IntPtr hWnd, int nIndex)
        {
            if (IntPtr.Size == 8)
                return GetWindowLongPtr64(hWnd, nIndex);
            else
                return GetWindowLongPtr32(hWnd, nIndex);
        }

        public static IntPtr SetWindowLongPtr(IntPtr hWnd, int nIndex, IntPtr dwNewLong)
        {
            if (IntPtr.Size == 8)
                return SetWindowLongPtr64(hWnd, nIndex, dwNewLong);
            else
                return new IntPtr(SetWindowLong32(hWnd, nIndex, dwNewLong.ToInt32()));
        }

        [DllImport("user32.dll")]
        static extern IntPtr GetMenu(IntPtr hWnd);

        [DllImport("user32.dll")]
        static extern IntPtr SetMenu(IntPtr hWnd, IntPtr hMenu);

        [DllImport("user32.dll", EntryPoint = "SetWindowLong", SetLastError = true)]
        private static extern int SetWindowLong32(IntPtr hWnd, int nIndex, int dwNewLong);

        [DllImport("user32.dll", EntryPoint = "SetWindowLongPtr", SetLastError = true)]
        private static extern IntPtr SetWindowLongPtr64(IntPtr hWnd, int nIndex, IntPtr dwNewLong);

        [DllImport("user32.dll", SetLastError = true)]
        static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int X, int Y, int cx, int cy, uint uFlags);

        [DllImport("user32.dll", SetLastError = true)]
        static extern bool LogicalToPhysicalPointForPerMonitorDPI(IntPtr hWnd, ref POINT lpPoint); //Win 8.1-10
        [DllImport("user32.dll", SetLastError = true)]
        static extern bool LogicalToPhysicalPoint(IntPtr hWnd, ref POINT lpPoint); //Win Vista-8

        [DllImport("user32.dll", SetLastError = true, CharSet = CharSet.Auto)]
        static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);

        [DllImport("user32.dll", SetLastError = true, CharSet = CharSet.Auto)]
        static extern bool SetWindowText(IntPtr hwnd, String lpString);

        [DllImport("user32.dll", CharSet = CharSet.Auto)]
        static extern IntPtr SendMessage(IntPtr hWnd, int Msg, int wParam, IntPtr lParam);

        [DllImport("user32.dll", CharSet = CharSet.Auto)]
        static extern IntPtr SendMessage(IntPtr hWnd, int Msg, IntPtr wParam, StringBuilder lParam);

        private static int WM_GETTEXTLENGTH = 0x000E;
        private static int WM_GETTEXT = 0x000D;
        private static string SendMessageGetWindowText(IntPtr hwnd)
        {
            int textLength = (int)SendMessage(hwnd, WM_GETTEXTLENGTH, 0, IntPtr.Zero);
            StringBuilder builder = new StringBuilder(textLength + 1);
            SendMessage(hwnd, WM_GETTEXT, (IntPtr)builder.Capacity, builder);
            return builder.ToString();
        }

        [DllImport("user32.dll", SetLastError = true, CharSet = CharSet.Auto)]
        static extern int GetClassName(IntPtr hWnd, StringBuilder lpClassName, int nMaxCount);
        [DllImport("user32.dll", SetLastError = true)]
        static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);
        private delegate bool EnumWindow(IntPtr hWnd, IntPtr lParam);
        [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern bool EnumWindows(EnumWindow lpEnumFunc, IntPtr lParam);
        private static IntPtr GetProcessMainWindow()
        {
            return GetProcessWindow("UnityContainerWndClass", "Unity", false);
        }
        private static IntPtr GetProcessWindow(EditorWindow editorWindow)
        {
            IntPtr windowHandle;
            string windowClass = "UnityPopupWndClass";
            bool fullscreenContainer = editorWindow.GetShowMode() != EditorWindowExtensions.ShowMode.PopupMenu;
            if (fullscreenContainer)
            {
                windowClass = "UnityContainerWndClass";
            }

            windowHandle = GetProcessWindow(windowClass, editorWindow);

            if (windowHandle == IntPtr.Zero)
            {
                //Try again but search for any class
                windowHandle = GetProcessWindow(null, editorWindow);
            }

            return windowHandle;
        }
        private static IntPtr GetProcessWindow(string withClassName, string withTitleMatching, bool fullTitleMatch)
        {
            return GetProcessWindow(withClassName, withTitleMatching, new Rect(0, 0, 0, 0), fullTitleMatch);
        }
        private static IntPtr GetProcessWindow(string withClassName, EditorWindow editorWindow)
        {
            EWFDebugging.StartTimer("Getting Process Window");
            IntPtr foundWindow = IntPtr.Zero;
            string searchTitle = editorWindow.GetIdentifierTitle();

            var windowSearchPos = editorWindow.GetContainerPosition(true);

            foundWindow = GetProcessWindow(withClassName, searchTitle, windowSearchPos, true);

            //Find window only by title if couldn't find by position and title
            if (foundWindow == IntPtr.Zero)
                foundWindow = GetProcessWindow(withClassName, searchTitle, new Rect(0, 0, 0, 0), true);

            //Find window only by position if couldn't find by title
            if (foundWindow == IntPtr.Zero)
                foundWindow = GetProcessWindow(withClassName, null, windowSearchPos, true);

            EWFDebugging.LogLine("--Found EditorWindow: " + (foundWindow != IntPtr.Zero) + " with title: '" + searchTitle + "' (" + editorWindow.GetWindowType() + ")" + ", position: " + windowSearchPos);
            EWFDebugging.LogTime("Getting Process Window");
            return foundWindow;
        }
        /// <summary>
        /// Get the handle to a window within the process using the following search parameters.
        /// </summary>
        /// <param name="withClassName">Only matches windows with this class name. If null, ignores the class name.</param>
        /// <param name="withTitleMatching">Only matches windows with this title. If null, ignores the title.</param>
        /// <param name="withRectMatching">Only matches windows which match this Rect position. If a zero rect, ignores this param.</param>
        /// <param name="fullTitleMatch">True if a full title match is required, or false if only need a partial match.</param>
        /// <returns></returns>
        private static IntPtr GetProcessWindow(string withClassName, string withTitleMatching, Rect withRectMatching, bool fullTitleMatch)
        {
            IntPtr hFoundWindow = IntPtr.Zero;
            int processID = System.Diagnostics.Process.GetCurrentProcess().Id;
            bool foundAMatch = false;
            float bestCloseness = Mathf.Infinity;
            string logInfo = "";
            logInfo += ("-----Looping through process windows. Process ID: " + processID + ". Looking for window: " + withClassName + ":" + (withTitleMatching == null ? "null" : withTitleMatching) + " fullTitleMatch: " + fullTitleMatch) + "\n";
            EnumWindows(
                delegate (IntPtr windowHandle, IntPtr lParam)
                {
                    uint winProcessID;
                    GetWindowThreadProcessId(windowHandle, out winProcessID);
                    LogWin32Error("Error getting process ID");

                    if (winProcessID == processID)
                    {
                        StringBuilder className = new StringBuilder(256);

                        GetClassName(windowHandle, className, className.Capacity);
                        LogWin32Error("Error getting class name");

                        if (withClassName == null || className.ToString() == withClassName)
                        {
                            string titleText = SendMessageGetWindowText(windowHandle);
                            if (EWFDebugging.Enabled)
                            {
                                LogWin32Error("Error getting window title text");
                            }
                            logInfo += ("-----  Window Class Name: '" + className.ToString() + "'. Window Title: '" + titleText) + "'\n";

                            if (withTitleMatching != null && ((!fullTitleMatch && titleText.Contains(withTitleMatching)) || titleText == withTitleMatching) ||
                                withTitleMatching == null && !withRectMatching.IsZero())
                            {
                                Rect windowBounds = new Rect(0, 0, 0, 0);
                                bool windowPositionMatch = true;
                                bool windowBoundsExactMatch = true;
                                float windowPosCloseness = 0;

                                //If withRectMatching is 0 then don't check it.
                                if (!withRectMatching.IsZero())
                                {
                                    windowBounds = GetWindowBounds(windowHandle);
                                    windowPositionMatch = windowBounds.x == withRectMatching.x && windowBounds.y == withRectMatching.y;
                                    windowBoundsExactMatch = windowBounds == withRectMatching;
                                    windowPosCloseness = windowBounds.Closeness(withRectMatching);
                                    logInfo += "-----    Comparing window bounds: " + windowBounds + " with rect: " + withRectMatching + "\n";
                                    logInfo += "-----    Position Match: " + windowPositionMatch + ", Bounds Exact Match: " + (windowBounds == withRectMatching) + ", Closeness: " + windowPosCloseness + "\n";
                                }
                                if (windowPositionMatch || windowPosCloseness < 75)
                                {
                                    foundAMatch = true;
                                    if (withTitleMatching == "Unity" && windowPositionMatch)
                                    {
                                        //Looking for the main Unity window
                                        hFoundWindow = windowHandle;

                                        //If the window title contains the Unity version, this must be the main window, so exit enumeration. Otherwise, get the last one (least z-index) which contains "Unity".
                                        if (titleText.ToString().Contains(Application.unityVersion))
                                        {
                                            logInfo += "-----    Found main window matching '*Unity*, *unityVersion*'\n";
                                            return false;
                                        }
                                    }
                                    else
                                    {
                                        //Find the closest match window size
                                        if (windowBoundsExactMatch || windowPosCloseness < bestCloseness)
                                        {
                                            hFoundWindow = windowHandle;
                                            bestCloseness = windowPosCloseness;
                                        }
                                        if (windowBoundsExactMatch)
                                        {
                                            return false; //Found an exact bounds match, so stop searching.
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if (withTitleMatching == "Unity" && String.IsNullOrEmpty(titleText.ToString()))
                                {
                                    if (!foundAMatch)
                                    {
                                        //Get the last window (least z-index). Workaround for the bug where main window can't be found on some systems because GetWindowText returns nothing.
                                        hFoundWindow = windowHandle;
                                    }
                                }
                            }

                        }
                    }
                    return true;
                }, IntPtr.Zero);
            LogWin32Error("Error when finding window of class: '" + withClassName + "', and title '" + withTitleMatching + "' Looking for full match: " + fullTitleMatch);
            logInfo += "-----Found window: " + (hFoundWindow != IntPtr.Zero) + ", handle: " + hFoundWindow.ToString();
            EWFDebugging.LogLine(logInfo, 0, 4);
            return hFoundWindow;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct RectStruct
        {
            public int Left;
            public int Top;
            public int Right;
            public int Bottom;
        }

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto)]
        internal struct MonitorInfoEx
        {
            public int Size; //Must be set to the byte size of this structure (Use Marshal.SizeOf()), so that GetMonitorInfo knows which struct is being passed.
            public RectStruct MonitorBounds; //Monitor bounds on the virtual screen
            public RectStruct WorkAreaBounds; //Work Area of the monitor (The bounds which a window maximizes to)
            public uint Flags; //If this value is 1, the monitor is the primary display
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = CCHDEVICENAME)]
            public string DeviceName; //The device name of the monitor

            public static MonitorInfoEx CreateWithDefaults()
            {
                var mi = new MonitorInfoEx();
                mi.DeviceName = String.Empty;
                mi.Size = Marshal.SizeOf(mi);
                return mi;
            }
        }

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto)]
        public struct Display_Device
        {
            [MarshalAs(UnmanagedType.U4)]
            public int Size;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)]
            public string DeviceName;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
            public string DeviceString;
            [MarshalAs(UnmanagedType.U4)]
            public DisplayDeviceStateFlags StateFlags;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
            public string DeviceID;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
            public string DeviceKey;

            public static Display_Device CreateWithDefaults()
            {
                var dd = new Display_Device();
                dd.Size = Marshal.SizeOf(dd);
                return dd;
            }
        }

        [Flags()]
        public enum DisplayDeviceStateFlags : int
        {
            AttachedToDesktop = 0x1,
            MultiDriver = 0x2,
            PrimaryDevice = 0x4,
            MirroringDriver = 0x8, //Represents a pseudo device used to mirror application drawing for remoting or other purposes. An invisible pseudo monitor is associated with this device.
            VGACompatible = 0x10,
            Removable = 0x20,
            ModesPruned = 0x8000000,
            Remote = 0x4000000,
            Disconnect = 0x2000000
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct DEVMODE
        {
            private const int CCHDEVICENAME = 0x20;
            private const int CCHFORMNAME = 0x20;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 0x20)]
            public string dmDeviceName;
            public short dmSpecVersion;
            public short dmDriverVersion;
            public short dmSize;
            public short dmDriverExtra;
            public int dmFields;
            public int dmPositionX;
            public int dmPositionY;
            public ScreenOrientation dmDisplayOrientation;
            public int dmDisplayFixedOutput;
            public short dmColor;
            public short dmDuplex;
            public short dmYResolution;
            public short dmTTOption;
            public short dmCollate;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 0x20)]
            public string dmFormName;
            public short dmLogPixels;
            public int dmBitsPerPel;
            public int dmPelsWidth;
            public int dmPelsHeight;
            public int dmDisplayFlags;
            public int dmDisplayFrequency;
            public int dmICMMethod;
            public int dmICMIntent;
            public int dmMediaType;
            public int dmDitherType;
            public int dmReserved1;
            public int dmReserved2;
            public int dmPanningWidth;
            public int dmPanningHeight;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct POINT
        {
            public int X;
            public int Y;

            public POINT(int x, int y)
            {
                this.X = x;
                this.Y = y;
            }
        }

        //Constants for use with SetWindowPos
#pragma warning disable 0414
        static readonly IntPtr HWND_TOPMOST = new IntPtr(-1);
        static readonly IntPtr HWND_NOTOPMOST = new IntPtr(-2);
        static readonly IntPtr HWND_TOP = new IntPtr(0);
        static readonly IntPtr HWND_BOTTOM = new IntPtr(1);
#pragma warning restore 0414

        /// <summary>
        /// Window handles (HWND) used for hWndInsertAfter
        /// </summary>
        public static class HWND
        {
            public static IntPtr
            NoTopMost = new IntPtr(-2),
            TopMost = new IntPtr(-1),
            Top = new IntPtr(0),
            Bottom = new IntPtr(1);
        }

        /// <summary>
        /// SetWindowPos Flags
        /// </summary>
        public struct SWP
        {
            public static readonly uint
            NOSIZE = 0x0001,
            NOMOVE = 0x0002,
            NOZORDER = 0x0004,
            NOREDRAW = 0x0008,
            NOACTIVATE = 0x0010,
            DRAWFRAME = 0x0020,
            FRAMECHANGED = 0x0020,
            SHOWWINDOW = 0x0040,
            HIDEWINDOW = 0x0080,
            NOCOPYBITS = 0x0100,
            NOOWNERZORDER = 0x0200,
            NOREPOSITION = 0x0200,
            NOSENDCHANGING = 0x0400,
            DEFERERASE = 0x2000,
            ASYNCWINDOWPOS = 0x4000;
        }

        const int GWL_HWNDPARENT = (-8);
        const int GWL_ID = (-12);
        const int GWL_STYLE = (-16);
        const int GWL_EXSTYLE = (-20);

        // Window Styles 
        const UInt32 WS_OVERLAPPED = 0;
        const UInt32 WS_POPUP = 0x80000000;
        const UInt32 WS_CHILD = 0x40000000;
        const UInt32 WS_MINIMIZE = 0x20000000;
        const UInt32 WS_VISIBLE = 0x10000000;
        const UInt32 WS_DISABLED = 0x8000000;
        const UInt32 WS_CLIPSIBLINGS = 0x4000000;
        const UInt32 WS_CLIPCHILDREN = 0x2000000;
        const UInt32 WS_MAXIMIZE = 0x1000000;
        const UInt32 WS_CAPTION = 0xC00000; //WS_BORDER or WS_DLGFRAME  
        const UInt32 WS_BORDER = 0x800000;
        const UInt32 WS_DLGFRAME = 0x400000;
        const UInt32 WS_VSCROLL = 0x200000;
        const UInt32 WS_HSCROLL = 0x100000;
        const UInt32 WS_SYSMENU = 0x80000;
        const UInt32 WS_THICKFRAME = 0x40000;
        const UInt32 WS_GROUP = 0x20000;
        const UInt32 WS_TABSTOP = 0x10000;
        const UInt32 WS_MINIMIZEBOX = 0x20000;
        const UInt32 WS_MAXIMIZEBOX = 0x10000;
        const UInt32 WS_TILED = WS_OVERLAPPED;
        const UInt32 WS_ICONIC = WS_MINIMIZE;
        const UInt32 WS_SIZEBOX = WS_THICKFRAME;

        // Extended Window Styles 
        const UInt32 WS_EX_DLGMODALFRAME = 0x0001;
        const UInt32 WS_EX_NOPARENTNOTIFY = 0x0004;
        const UInt32 WS_EX_TOPMOST = 0x0008;
        const UInt32 WS_EX_ACCEPTFILES = 0x0010;
        const UInt32 WS_EX_TRANSPARENT = 0x0020;
        const UInt32 WS_EX_MDICHILD = 0x0040;
        const UInt32 WS_EX_TOOLWINDOW = 0x0080;
        const UInt32 WS_EX_WINDOWEDGE = 0x0100;
        const UInt32 WS_EX_CLIENTEDGE = 0x0200;
        const UInt32 WS_EX_CONTEXTHELP = 0x0400;
        const UInt32 WS_EX_RIGHT = 0x1000;
        const UInt32 WS_EX_LEFT = 0x0000;
        const UInt32 WS_EX_RTLREADING = 0x2000;
        const UInt32 WS_EX_LTRREADING = 0x0000;
        const UInt32 WS_EX_LEFTSCROLLBAR = 0x4000;
        const UInt32 WS_EX_RIGHTSCROLLBAR = 0x0000;
        const UInt32 WS_EX_CONTROLPARENT = 0x10000;
        const UInt32 WS_EX_STATICEDGE = 0x20000;
        const UInt32 WS_EX_APPWINDOW = 0x40000;
        const UInt32 WS_EX_OVERLAPPEDWINDOW = (WS_EX_WINDOWEDGE | WS_EX_CLIENTEDGE);
        const UInt32 WS_EX_PALETTEWINDOW = (WS_EX_WINDOWEDGE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST);
        const UInt32 WS_EX_LAYERED = 0x00080000;
        const UInt32 WS_EX_NOINHERITLAYOUT = 0x00100000;
        const UInt32 WS_EX_LAYOUTRTL = 0x00400000;
        const UInt32 WS_EX_COMPOSITED = 0x02000000;
        const UInt32 WS_EX_NOACTIVATE = 0x08000000;
#endif
    }
#if UNITY_EDITOR_WIN
    internal static void SetMainWindowFullscreenStyle(EditorFullscreenState.WindowFullscreenState fullscreenState, bool showMenu)
    {
        WindowsDisplay.SetMainWindowFullscreenStyle(fullscreenState, showMenu);
    }

    internal static void SaveMainWindowStyleInState(EditorFullscreenState.WindowFullscreenState fullscreenState)
    {
        WindowsDisplay.SaveMainWindowStyleInState(fullscreenState);
    }

    internal static void LoadMainWindowStyleInState(EditorFullscreenState.WindowFullscreenState fullscreenState, bool makeResizable)
    {
        WindowsDisplay.LoadMainWindowStyleInState(fullscreenState, makeResizable);
    }
#endif
}