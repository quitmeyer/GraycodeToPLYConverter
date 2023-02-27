/* 
 * Author:  Johanan Round
 */

using System;
using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.Reflection;
using EditorWindowFullscreen;

/// <summary>
/// Get information about a system display. Recognize multiple displays and their positions on the screen.
/// Multidisplay support for Windows and Mac.
/// </summary>
public partial class SystemDisplay
{
    public string Name { get; private set; }

    public bool AttachedToDesktop { get; private set; }
    public bool IsPrimary { get; private set; }
    public bool HasMainWindow { get; private set; }

    public Rect Bounds { get; private set; }
    public Rect PhysicalBounds { get; private set; }
    public Rect WorkArea { get; private set; }

    public int PixelWidth { get; private set; }
    public int PixelHeight { get; private set; }

#if UNITY_EDITOR_WIN
    private class NativeDisplay : WindowsDisplay { }
#elif UNITY_EDITOR_OSX
    private class NativeDisplay : MacOSDisplay { }
#elif UNITY_EDITOR_LINUX
    private class NativeDisplay : LinuxDisplay { }
#else
    private class NativeDisplay {public static List<SystemDisplay> GetAllDisplays() {return null;}} //Fallback to single screen on any other platform
#endif

    public SystemDisplay()
    {
        this.Name = "";
    }

    private static SystemDisplay[] allDisplays; //Cache of all displays.

    /// <summary> All of the system displays. The displays are cached the first time they are retrieved.</summary>
    public static SystemDisplay[] AllDisplays
    {
        get
        {
            return allDisplays == null ? GetAllDisplays() : allDisplays;
        }
    }

    ///Get all the displays which are attached to the desktop
    public static SystemDisplay[] GetAllDisplays()
    {
        return GetAllDisplays(false);
    }
    ///Get all the displays, and choose whether to include monitors not attached to the desktop
    public static SystemDisplay[] GetAllDisplays(bool IncludeMonitorsNotAttachedToDesktop)
    {
        List<SystemDisplay> allDisplays = null;
        try
        {
            allDisplays = NativeDisplay.GetAllDisplays();
        }
        catch (System.Exception e)
        {
            Debug.LogError(e);
        }

        if (allDisplays == null || allDisplays.Count == 0)
        {
            /*Failed to find the displays, so add the primary Screen as a display*/
            var display = new SystemDisplay();
            display.Bounds = new Rect(0, 0, Screen.currentResolution.width, Screen.currentResolution.height);
            display.PhysicalBounds = display.Bounds;
            display.PixelWidth = (int)display.PhysicalBounds.width;
            display.PixelHeight = (int)display.PhysicalBounds.height;
            display.WorkArea = display.Bounds;
            display.AttachedToDesktop = true;
            display.IsPrimary = true;
            if (allDisplays == null)
            {
                allDisplays = new List<SystemDisplay>();
            }
            allDisplays.Add(display);
        }

        if (!IncludeMonitorsNotAttachedToDesktop)
        {
            //Remove displays not attached to the desktop
            allDisplays.RemoveAll(display => !display.AttachedToDesktop);
        }

        SystemDisplay.allDisplays = allDisplays.ToArray();
        return SystemDisplay.allDisplays;
    }

    /// Get the system display which contains the specified (x, y) position. Returns null if none of the displays contain the point.
    public static SystemDisplay ContainingPoint(int x, int y)
    {
        return AllDisplays == null ? null : AllDisplays.ContainingPoint(x, y);
    }

    /// Get the system display containing or closest to the specified (x, y) position.
    public static SystemDisplay ClosestToPoint(int x, int y)
    {
        return AllDisplays == null ? null : AllDisplays.ClosestToPoint(x, y);
    }
    /// Get the system display containing or closest to the specified point.
    public static SystemDisplay ClosestToPoint(Vector2 point)
    {
        return AllDisplays == null ? null : AllDisplays.ClosestToPoint(point);
    }

    /// Get the system display which has the main window
    public static SystemDisplay WithMainWindow()
    {
        return AllDisplays == null ? null : AllDisplays.WithMainWindow();
    }

#if UNITY_EDITOR_WIN
    public static bool WindowIsFullscreenOnDisplay(EditorWindow editorWindow, string windowTitle, SystemDisplay display)
    {
        return WindowsDisplay.WindowIsFullscreenOnDisplay(editorWindow, windowTitle, display);
    }
    /// Makes an editor window fullscreen on a system display.
    public static void MakeWindowCoverTaskBar(EditorWindow editorWindow, SystemDisplay display)
    {
        WindowsDisplay.MakeWindowCoverTaskBar(editorWindow, display);
    }
    /// Makes a window with the specified title fullscreen on a system display.
    public static void MakeWindowCoverTaskBar(string windowClass, string windowTitle, SystemDisplay display)
    {
        WindowsDisplay.MakeWindowCoverTaskBar(windowClass, windowTitle, display);
    }
#endif

#if UNITY_EDITOR_OSX
    /// Sets the position of a window using system calls.
    public static void SetWindowPosition(EditorWindow editorWindow, int x, int y, int width, int height)
    {
        UseIdentifierTitle(editorWindow, () => NativeDisplay.SetWindowPosition(editorWindow, x, y, width, height));
    }
#endif

    /// Restores the style of a window to what it was before fullscreening.
    public static void RestoreWindowStyle(EditorWindow editorWindow, Rect origPosition)
    {
        UseIdentifierTitle(editorWindow, () => NativeDisplay.RestoreWindowStyle(editorWindow, origPosition));
    }

    ///Use an identifier title on the window so that it can be identified in system calls
    private static void UseIdentifierTitle(EditorWindow editorWindow, Action actionToPerform)
    {
        if (editorWindow == null) return;
        string prevIdentifierTitle = editorWindow.GetIdentifierTitle();
        editorWindow.SetIdentifierTitle();
        actionToPerform.Invoke();
        editorWindow.SetIdentifierTitle(prevIdentifierTitle);
    }

    /// Converts a logical point to a physical point
    public static Vector2 LogicalToPhysicalPoint(Vector2 logicalPoint)
    {
        Vector2 physPoint;
        MethodInfo getPhysicalPoint = null;
        try
        {
            getPhysicalPoint = typeof(NativeDisplay).BaseType.GetMethod("GetPhysicalPoint", BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic, null, new[] { typeof(Vector2) }, null);
        }
        catch
        {
            Debug.Log("error finding physical point.");
        }

        //If the OS Native Display class has the method, call that, otherwise use a fallback.
        if (getPhysicalPoint != null)
        {
            physPoint = (Vector2)getPhysicalPoint.Invoke(null, new object[] { logicalPoint });
        }
        else
        {
            physPoint = logicalPoint;
        }

        return physPoint;
    }
}

public static class SystemDisplayExtensions
{
    /// Returns true if the display contains the specified logical point (logical point differs from physical point when there is display scaling)
    public static bool ContainsPoint(this SystemDisplay display, Vector2 logicalPoint)
    {
        return (display.Bounds.Contains(logicalPoint));
    }
    public static bool ContainsPoint(this SystemDisplay display, Vector2 point, bool physicalPoint)
    {
        if (physicalPoint)
        {
            return display.PhysicalBounds.Contains(point);
        }
        else
        {
            return (display.Bounds.Contains(point));
        }
    }

    /// Get the system display within the array which contains the specified (x, y) position. 
    public static SystemDisplay ContainingPoint(this SystemDisplay[] displayList, int x, int y)
    {
        var physicalPoint = new Vector2(x, y);
        return displayList.ContainingPoint(physicalPoint);
    }

    /// Get the system display within the array which contains the specified point. Returns null if none of the displays contain the point.
    public static SystemDisplay ContainingPoint(this SystemDisplay[] displayList, Vector2 logicalPoint)
    {
        return displayList.ContainingPoint(logicalPoint, false);
    }
    public static SystemDisplay ContainingPoint(this SystemDisplay[] displayList, Vector2 point, bool physicalPoint)
    {
        foreach (SystemDisplay display in displayList)
        {
            if (display.ContainsPoint(point, physicalPoint)) return display;
        }
        return null;
    }

    /// Get the system display within the array which is containing or closest to the specified (x, y) position.
    public static SystemDisplay ClosestToPoint(this SystemDisplay[] displayList, int x, int y)
    {
        return ClosestToPoint(displayList, new Vector2(x, y));
    }

    /// Get the system display within the array which is containing or closest to the specified point.
    public static SystemDisplay ClosestToPoint(this SystemDisplay[] displayList, Vector2 logicalPoint)
    {
        return displayList.ClosestToPoint(logicalPoint, false);
    }
    public static SystemDisplay ClosestToPoint(this SystemDisplay[] displayList, Vector2 point, bool physicalPoint)
    {
        float closestDistance = 0;
        SystemDisplay closestDisplay = null;

        foreach (SystemDisplay display in displayList)
        {
            if (display.ContainsPoint(point, physicalPoint)) return display;

            var dist = physicalPoint ? display.PhysicalBounds.DistanceToPoint(point) : display.Bounds.DistanceToPoint(point);
            if (dist < closestDistance || closestDisplay == null)
            {
                closestDistance = dist;
                closestDisplay = display;
            }
        }

        return closestDisplay;
    }

    /// Get the system display within the array which has the main window
    public static SystemDisplay WithMainWindow(this SystemDisplay[] displayList)
    {
        foreach (SystemDisplay display in displayList)
        {
            if (display.HasMainWindow) return display;
        }
        return null;
    }
}


