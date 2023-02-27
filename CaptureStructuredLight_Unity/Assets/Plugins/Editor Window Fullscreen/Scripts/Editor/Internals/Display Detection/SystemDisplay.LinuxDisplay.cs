using UnityEngine;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;


public partial class SystemDisplay
{
    /// <summary>
    /// Linux-specific methods of SystemDisplay
    /// </summary>
    private class LinuxDisplay
    {
        public static List<SystemDisplay> GetAllDisplays()
        {
            return null;
        }
    }
}
