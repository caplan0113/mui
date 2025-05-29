# Memo

```C#
[StructLayout(LayoutKind.Sequential)]
public struct XrSingleEyeGazeDataHTC
{
    /// <summary>
    /// An <see cref="XrBool32">XrBool32</see> indicating if the returned gazePose is valid. Callers should check the validity of pose prior to use.
    /// </summary>
    public XrBool32 isValid;
    /// <summary>
    /// An <see cref="XrPosef">XrPosef</see> describing the position and orientation of the user's eye. The pose is represented in the coordinate system provided by <see cref="XrEyeGazeDataInfoHTC">XrEyeGazeDataInfoHTC</see>::<see cref="XrEyeGazeDataInfoHTC.baseSpace">baseSpace</see>.
    /// </summary>
    public XrPosef gazePose;

    /// <param name="in_isValid">An <see cref="XrBool32">XrBool32</see> indicating if the returned gazePose is valid. Callers should check the validity of pose prior to use.</param>
    /// <param name="in_gazePose">An <see cref="XrPosef">XrPosef</see> describing the position and orientation of the user's eye. The pose is represented in the coordinate system provided by <see cref="XrEyeGazeDataInfoHTC">XrEyeGazeDataInfoHTC</see>::<see cref="XrEyeGazeDataInfoHTC.baseSpace">baseSpace</see>.</param>
    public XrSingleEyeGazeDataHTC(XrBool32 in_isValid, XrPosef in_gazePose)
    {
        isValid = in_isValid;
        gazePose = in_gazePose;
    }
};
```

```C#
[StructLayout(LayoutKind.Sequential)]
public struct XrSingleEyePupilDataHTC
{
    /// <summary>
    /// An <see cref="XrBool32">XrBool32</see> indicating if the returned pupilDiameter is valid. Callers should check the validity of diameter prior to use.
    /// </summary>
    public XrBool32 isDiameterValid;
    /// <summary>
    /// An <see cref="XrBool32">XrBool32</see> indicating if the returned pupilPosition is valid. Callers should check the validity of position prior to use.
    /// </summary>
    public XrBool32 isPositionValid;
    /// <summary>
    /// The diameter of pupil in millimeters.
    /// </summary>
    public float pupilDiameter;
    /// <summary>
    /// The position of pupil in sensor area which x and y are normalized in [0,1] with +Y up and +X to the right.
    /// </summary>
    public XrVector2f pupilPosition;

    /// <param name="in_isDiameterValid">An <see cref="XrBool32">XrBool32</see> indicating if the returned gazePose is valid. Callers should check the validity of pose prior to use.</param>
    /// <param name="in_isPositionValid">An <see cref="XrBool32">XrBool32</see> indicating if the returned pupilPosition is valid. Callers should check the validity of position prior to use.</param>
    /// <param name="in_pupilDiameter">The diameter of pupil in millimeters.</param>
    /// <param name="in_pupilPosition">The position of pupil in sensor area which x and y are normalized in [0,1]with +Y up and +X to the right.</param>
    public XrSingleEyePupilDataHTC(XrBool32 in_isDiameterValid, XrBool32 in_isPositionValid, float in_pupilDiameter, XrVector2f in_pupilPosition)
    {
        isDiameterValid = in_isDiameterValid;
        isPositionValid = in_isPositionValid;
        pupilDiameter  = in_pupilDiameter;
        pupilPosition  = in_pupilPosition;
    }
};
```

```C#
[StructLayout(LayoutKind.Sequential)]
public struct XrSingleEyeGeometricDataHTC
{
    /// <summary>
    /// A flag that indicates if the geometric data is valid. Callers should check the validity of the geometric data prior to use.
    /// </summary>
    public XrBool32 isValid;
    /// <summary>
    /// A value in range [0,1] representing the openness of the user's eye. When this value is zero, the eye closes normally. When this value is one, the eye opens normally. When this value goes higher, the eye approaches open.
    /// </summary>
    public float eyeOpenness;
    /// <summary>
    /// A value in range [0,1] representing how the user's eye open widely. When this value is zero, the eye opens normally. When this value goes higher, the eye opens wider.
    public float eyeWide;
    /// <summary>
    /// A value in range [0,1] representing how the user's eye is closed. When this value is zero, the eye closes normally. When this value goes higher, the eye closes tighter.
    /// </summary>
    public float eyeSqueeze;

    /// <param name="in_isValid">A flag that indicates if the geometric data is valid. Callers should check the validity of the geometric data prior to use.</param>
    /// <param name="in_eyeOpenness">A value in range [0,1] representing the openness of the user's eye. When this value is zero, the eye closes normally. When this value is one, the eye opens normally. When this value goes higher, the eye approaches open.</param>
    /// <param name="in_eyeWide">A value in range [0,1] representing how the user's eye open widely. When this value is zero, the eye opens normally. When this value goes higher, the eye opens wider.</param>
    /// <param name="in_eyeSqueeze">A value in range [0,1] representing how the user's eye is closed. When this value is zero, the eye closes normally. When this value goes higher, the eye closes tighter.</param>
    public XrSingleEyeGeometricDataHTC(XrBool32 in_isValid, float in_eyeOpenness, float in_eyeWide, float in_eyeSqueeze)
    {
        isValid = in_isValid;
        eyeOpenness = in_eyeOpenness;
        eyeWide = in_eyeWide;
        eyeSqueeze = in_eyeSqueeze;
    }
};
```
