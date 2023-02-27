/* 
 * Author:  Johanan Round
 * Package: Editor Window Fullscreen
 * License: Unity Asset Store EULA (Editor extension asset. Requires 1 license per machine.)
 */

using UnityEditor;
using UnityEngine;

using System;
using System.Reflection;
using System.Collections.Generic;

namespace EditorWindowFullscreen
{
    /// <summary>
    /// Get information on the displays while in the editor
    /// </summary>
    public sealed class EditorDisplay
    {

        public Rect Bounds { get; private set; }
        public bool PrimaryDisplay { get; private set; }

        private static System.Type ieuType;
        private static MethodInfo getBoundsOfDesktopAtPoint;

        private static List<Vector2> LockDisplayAtPoints = new List<Vector2>();

        /// <summary>
        /// Locked indicates that an operation is currently being performed on the display and we shouldn't do anything else on it until the lock is released.
        /// The locked status is persistent regardless of whether the cached displays are cleared.
        /// </summary>
        /// <value><c>true</c> if locked; otherwise, <c>false</c>.</value>
        public bool Locked
        {
            get { return LockDisplayAtPoints.Exists(p => this.Bounds.Contains(p)); }
            set 
            {
                if (value == true && !Locked) LockDisplayAtPoints.Add(this.Bounds.center);
                else if (value == false) LockDisplayAtPoints.RemoveAll(p => this.Bounds.Contains(p));
            }
        }

        static EditorDisplay()
        {
            ieuType = System.Type.GetType("UnityEditorInternal.InternalEditorUtility,UnityEditor");
            getBoundsOfDesktopAtPoint = ieuType.GetMethod("GetBoundsOfDesktopAtPoint", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static, null, new Type[] { typeof(Vector2) }, null);
        }

        public EditorDisplay(Rect bounds)
        {
            this.Bounds = bounds;
        }

        public static Rect PrimaryDesktopResolution
        {
            get
            {
                return new Rect(0, 0, Screen.currentResolution.width, Screen.currentResolution.height);
            }
        }

        private static List<EditorDisplay> allDisplays; //Cache of all displays.

        /// <summary> 
        /// All of the EditorDisplays. The displays are cached the first time they are retrieved.
        /// </summary>
        public static List<EditorDisplay> AllDisplays
        {
            get
            {
                return allDisplays == null ? GetAllDisplays() : allDisplays;
            }
        }

        /// <summary>
        /// Clear all the cached displays, so that they are refreshed on the next call to AllDisplays.
        /// </summary>
        public static void ClearCachedDisplays()
        {
            allDisplays = null;
        }

        /// <summary> 
        /// Get all the displays which are attached to the desktop (As a List)
        /// </summary>
        public static List<EditorDisplay> GetAllDisplays()
        {
            EWFDebugging.StartTimer("GetAllDisplays");
            List<EditorDisplay> allDisplays = new List<EditorDisplay>();

            try
            {
#if UNITY_EDITOR_OSX
                //Get system displays
                allDisplays = FromSystemDisplays(SystemDisplay.GetAllDisplays());
#else
                allDisplays = FromSystemDisplays(SystemDisplay.GetAllDisplays());

                //If couldn't find system displays, use backup method
                if (allDisplays == null || allDisplays.Count < 1)
                {
                    var desktopBounds = PrimaryDesktopResolution;
                    allDisplays = new List<EditorDisplay>();
                    //Find all the displays
                    var display = AddDisplayAtPoint(allDisplays, desktopBounds.center, true);
                    if (display != null)
                        AddContiguousDisplays(allDisplays, display);
                }
#endif
            }
            catch (Exception e)
            {
                if (EWFDebugging.Enabled)
                {
                    string err = "Failed to find all possible displays. " + e;
                    Debug.LogError(err);
                    EWFDebugging.LogError(err);
                }
            }

            if (allDisplays.Count == 0)
            {
                /*Failed to find the displays, so add the primary Screen as a display*/
                var display = new EditorDisplay(new Rect(0, 0, Screen.currentResolution.width, Screen.currentResolution.height));
                allDisplays.Add(display);
            }

            //Sort screens by top-left to bottom-right
            allDisplays.Sort(delegate (EditorDisplay a, EditorDisplay b)
            {
                bool aIsLess;
                if (a.Bounds.y != b.Bounds.y)
                    aIsLess = a.Bounds.y < b.Bounds.y;
                else
                {
                    if (a.Bounds.x == b.Bounds.x)
                        return 0; //Equal
                    else
                        aIsLess = a.Bounds.x < b.Bounds.x;
                }
                return aIsLess ? -1 : 1;
            });

            EWFDebugging.LogTime("GetAllDisplays", false);
            EditorDisplay.allDisplays = allDisplays;
            return allDisplays;
        }

        /// <summary> 
        /// Get an EditorDisplay list from a SystemDisplay array
        /// </summary>
        private static List<EditorDisplay> FromSystemDisplays(SystemDisplay[] sysDisplayList)
        {
            if (sysDisplayList == null) return null;
            var displayList = new List<EditorDisplay>();
            foreach (var sys in sysDisplayList)
            {
                displayList.Add(FromSystemDisplay(sys));
            }
            return displayList;
        }

        /// <summary> 
        /// Get an EditorDisplay from a SystemDisplay
        /// </summary>
        private static EditorDisplay FromSystemDisplay(SystemDisplay sys)
        {
            if (sys == null) return null;
            EditorDisplay ed = new EditorDisplay(sys.Bounds);
            ed.PrimaryDisplay = sys.IsPrimary;
            return ed;
        }

        /// <summary> 
        /// Get all the displays which are attached to the desktop (As an array)
        /// </summary>
        public static EditorDisplay[] GetAllDisplaysArray()
        {
            return GetAllDisplays().ToArray();
        }

        private static void AddContiguousDisplays(List<EditorDisplay> allDisplays, EditorDisplay display)
        {
            int searchDistance = 2700; //Other displays which are scaled may not be touching the display's bounds. So search a distance outside the display.
            int x, y, halfStep = 150, step = 300;
            EditorDisplay newDisplay;

            for (y = -searchDistance - halfStep; y < display.Bounds.height * 2 + step; y += step) //Display could be scaled up to 175% so check 2x height
            {
                newDisplay = AddDisplayAtPoint(allDisplays, new Vector2(-halfStep, y));
                if (newDisplay != null) AddContiguousDisplays(allDisplays, newDisplay);

                newDisplay = AddDisplayAtPoint(allDisplays, new Vector2(display.Bounds.width + halfStep, y));
                if (newDisplay != null) AddContiguousDisplays(allDisplays, newDisplay);
            }

            searchDistance = 5400;
            for (x = -searchDistance - halfStep; x < display.Bounds.width * 2 + step; x += step)
            {
                newDisplay = AddDisplayAtPoint(allDisplays, new Vector2(x, -halfStep));
                if (newDisplay != null) AddContiguousDisplays(allDisplays, newDisplay);

                newDisplay = AddDisplayAtPoint(allDisplays, new Vector2(x, display.Bounds.height + halfStep));
                if (newDisplay != null) AddContiguousDisplays(allDisplays, newDisplay);
            }
        }

        private static EditorDisplay AddDisplayAtPoint(List<EditorDisplay> allDisplays, Vector2 point)
        {
            return AddDisplayAtPoint(allDisplays, point, false);
        }
        private static EditorDisplay AddDisplayAtPoint(List<EditorDisplay> allDisplays, Vector2 point, bool primaryDisplay)
        {
            if (!allDisplays.Exists(d => d.Bounds.Contains(point)))
            {
                var displayBounds = GetBoundsOfDesktopAtPoint(point);
                if (!allDisplays.Exists(d => d.Bounds == displayBounds))
                {
                    var foundDisplay = new EditorDisplay(displayBounds);
                    foundDisplay.PrimaryDisplay = primaryDisplay;
                    allDisplays.Add(foundDisplay);
                    return foundDisplay;
                }
            }
            return null;
        }

        /// <summary>
        /// Get the desktop bounds of the display containing or closest to a point.
        /// </summary>
        public static Rect GetBoundsOfDesktopAtPoint(Vector2 point)
        {
            return (Rect)getBoundsOfDesktopAtPoint.Invoke(null, new object[] { point });
        }


        /// <summary>
        /// Get the display which contains the specified (x, y) position. Returns null if none of the displays contain the point.
        /// </summary>
        public static EditorDisplay ContainingPoint(int x, int y)
        {
            return ContainingPoint(new Vector2(x, y));
        }
        /// <summary>
        /// Get the display which contains the specified point. Returns null if none of the displays contain the point.
        /// </summary>
        public static EditorDisplay ContainingPoint(Vector2 point)
        {
            foreach (EditorDisplay display in AllDisplays)
            {
                if (display.Bounds.Contains(point)) return display;
            }

            return null;
        }
        /// <summary>
        /// Get the display containing or closest to the specified (x, y) position.
        /// </summary>
        public static EditorDisplay ClosestToPoint(int x, int y)
        {
            return ClosestToPoint(new Vector2(x, y));
        }
        /// <summary>
        /// Get the display containing or closest to the specified point.
        /// </summary>
        public static EditorDisplay ClosestToPoint(Vector2 point)
        {
            float closestDistance = 0;
            EditorDisplay closestDisplay = null;

            foreach (EditorDisplay display in AllDisplays)
            {
                if (display.Bounds.Contains(point)) return display;

                var dist = display.Bounds.DistanceToPoint(point);
                if (dist < closestDistance || closestDisplay == null)
                {
                    closestDistance = dist;
                    closestDisplay = display;
                }
            }

            return closestDisplay;
        }
    }
}