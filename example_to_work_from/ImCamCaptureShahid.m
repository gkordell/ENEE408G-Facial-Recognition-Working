function varargout = ImCamCaptureShahid(varargin)
% IMCAMCAPTURESHAHID MATLAB code for ImCamCaptureShahid.fig
%      IMCAMCAPTURESHAHID, by itself, creates a new IMCAMCAPTURESHAHID or raises the existing
%      singleton*.
%
%      H = IMCAMCAPTURESHAHID returns the handle to a new IMCAMCAPTURESHAHID or the handle to
%      the existing singleton*.
%
%      IMCAMCAPTURESHAHID('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in IMCAMCAPTURESHAHID.M with the given input arguments.
%
%      IMCAMCAPTURESHAHID('Property','Value',...) creates a new IMCAMCAPTURESHAHID or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before ImCamCaptureShahid_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to ImCamCaptureShahid_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help ImCamCaptureShahid

% Last Modified by GUIDE v2.5 17-Apr-2018 12:58:45

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ImCamCaptureShahid_OpeningFcn, ...
                   'gui_OutputFcn',  @ImCamCaptureShahid_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before ImCamCaptureShahid is made visible.
function ImCamCaptureShahid_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to ImCamCaptureShahid (see VARARGIN)

% Choose default command line output for ImCamCaptureShahid
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes ImCamCaptureShahid wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = ImCamCaptureShahid_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
