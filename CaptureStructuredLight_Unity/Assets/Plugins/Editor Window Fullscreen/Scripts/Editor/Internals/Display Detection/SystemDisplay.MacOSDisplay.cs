using CGFloat = System.IntPtr;
using UnityEditor;
using UnityEngine;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using EditorWindowFullscreen;

public partial class SystemDisplay
{
    /// <summary>
    /// Mac-specific methods of SystemDisplay
    /// </summary>
    protected class MacOSDisplay
    {
#if UNITY_EDITOR_OSX
        public static List<SystemDisplay> GetAllDisplays()
        {
            List<SystemDisplay> allDisplays = new List<SystemDisplay>();

            uint[] activeDisplays = new uint[32];
            uint displayCount = 0;
            CGGetActiveDisplayList(32, activeDisplays, ref displayCount);

            for (int i = 0; i < displayCount; i++)
            {
                var displayID = activeDisplays[i];
                var display = new SystemDisplay();
                display.AttachedToDesktop = true;
                display.IsPrimary = CGDisplayIsMain(displayID);
                display.HasMainWindow = display.IsPrimary;
                display.PixelHeight = (int)CGDisplayPixelsHigh(displayID);
                display.PixelWidth = (int)CGDisplayPixelsWide(displayID);
                var dispBounds = CGDisplayBounds(displayID);
                var position = CGPoint.ToVector2(dispBounds.origin);
                var size = CGSize.ToVector2(dispBounds.size);
                var bounds = new Rect(position.x, position.y, size.x, size.y);

                display.Bounds = bounds;
                display.PhysicalBounds = display.Bounds;
                display.WorkArea = display.Bounds;

                allDisplays.Add(display);
            }
            return allDisplays;
        }

        private static float FloatFromIntPtrBits(IntPtr ptr)
        {
            float number;
            byte[] bytes = BytesFromIntPtr(ptr);
            if (bytes.Length == 8)
            {
                //64 bit system
                number = (float)BitConverter.ToDouble(bytes, 0);
            }
            else
            {
                //32 bit system
                number = BitConverter.ToSingle(bytes, 0);
            }
            return number;
        }
        private static byte[] BytesFromIntPtr(IntPtr ptr)
        {
            byte[] bytes;
            if (IntPtr.Size == 8)
            {
                //64 bit system
                long num = (long)ptr;
                bytes = BitConverter.GetBytes(num);
            }
            else
            {
                //32 bit system
                int num = (int)ptr;
                bytes = BitConverter.GetBytes(num);
            }
            return bytes;
        }

#pragma warning disable 649
        public struct CGRect
        {
            public CGPoint origin;
            public CGSize size;
        }
        public struct CGSize
        {
            public CGFloat width;
            public CGFloat height;
            public static Vector2 ToVector2(CGSize size)
            {
                return new Vector2(FloatFromIntPtrBits(size.width), FloatFromIntPtrBits(size.height));
            }
        }
        public struct CGPoint
        {
            public CGFloat x;
            public CGFloat y;
            public static Vector2 ToVector2(CGPoint point)
            {
                return new Vector2(FloatFromIntPtrBits(point.x), FloatFromIntPtrBits(point.y));
            }
        }
#pragma warning restore 649

        [DllImport("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")]
        static extern int CGGetActiveDisplayList(uint maxDisplays, uint[] activeDisplays, ref uint displayCount);

        [DllImport("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")]
        static extern bool CGDisplayIsMain(uint cgDisplayID);

        [DllImport("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")]
        static extern CGRect CGDisplayBounds(uint cgDisplayID);

        [DllImport("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")]
        static extern IntPtr CGDisplayPixelsHigh(uint cgDisplayID);

        [DllImport("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")]
        static extern IntPtr CGDisplayPixelsWide(uint cgDisplayID);

        internal static void SetWindowPosition(EditorWindow editorWindow, int x, int y, int width, int height)
        {
            SetWindowPositionForID(editorWindow.GetIdentifierID(), x, y, width, height);
        }
        internal static void RestoreWindowStyle(EditorWindow editorWindow, Rect origPosition)
        {
            RestoreWindowStyleForID(editorWindow.GetIdentifierID(), (int)origPosition.x, (int)origPosition.y, (int)origPosition.width, (int)origPosition.height);
        }
#endif
    }

#if UNITY_EDITOR_OSX
    //EWFMac
    [DllImport("EWFMac")]
    public static extern void EnableDebugging(bool enable);

    [DllImport("EWFMac")]
    public static extern IntPtr GetMainWindowController();

    [DllImport("EWFMac")]
    public static extern IntPtr ToggleFullscreenMainWindow(int origX, int origY, int origWidth, int origHeight);

    [DllImport("EWFMac")]
    public static extern bool MainWindowIsFullscreen();

    [DllImport("EWFMac")]
    public static extern void RestoreMainWindowStyle();

    [DllImport("EWFMac")]
    public static extern IntPtr SetMainWindowPosition(int x, int y, int width, int height);

    [DllImport("EWFMac")]
    public static extern bool MainWindowIsExitingFullscreen();

    [DllImport("EWFMac")]
    public static extern bool FocusMainWindow();

    [DllImport("EWFMac")]
    public static extern bool CloseWindow(long identifierID);

    [DllImport("EWFMac")]
    public static extern bool SetWindowPositionForID(long identifierID, int x, int y, int width, int height);

    [DllImport("EWFMac")]
    public static extern bool RestoreWindowStyleForID(long identifierID, int origPosX, int origPosY, int origWidth, int origHeight);

    [DllImport("EWFMac")]
    public static extern bool WindowIsFullscreenWithID(long identifierID);

    [DllImport("EWFMac")]
    public static extern bool FullscreenWindowWithID(long identifierID);

#endif
}
