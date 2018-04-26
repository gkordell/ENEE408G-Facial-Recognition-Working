function varargout = really_basic_example_2(varargin)
% REALLY_BASIC_EXAMPLE MATLAB code for really_basic_example.fig
%      REALLY_BASIC_EXAMPLE, by itself, creates a new REALLY_BASIC_EXAMPLE or raises the existing
%      singleton*.
%
%      H = REALLY_BASIC_EXAMPLE returns the handle to a new REALLY_BASIC_EXAMPLE or the handle to
%      the existing singleton*.
%
%      REALLY_BASIC_EXAMPLE('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in REALLY_BASIC_EXAMPLE.M with the given input arguments.
%
%      REALLY_BASIC_EXAMPLE('Property','Value',...) creates a new REALLY_BASIC_EXAMPLE or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before really_basic_example_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to really_basic_example_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help really_basic_example

% Last Modified by GUIDE v2.5 24-Apr-2018 12:52:17

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @really_basic_example_OpeningFcn, ...
                   'gui_OutputFcn',  @really_basic_example_OutputFcn, ...
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
end

% --- Executes just before really_basic_example is made visible.
function really_basic_example_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to really_basic_example (see VARARGIN)

% Choose default command line output for really_basic_example
global vid
global count = 0;
handles.output = hObject;
axes(handles.axes1);
vid = videoinput('winvideo', 1, 'YUY2_640x480');
hImage = image(zeros(480,640,3), 'Parent',handles.axes1);
preview(vid, hImage);

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes really_basic_example wait for user response (see UIRESUME)
% uiwait(handles.figure1);
end

% --- Outputs from this function are returned to the command line.
function varargout = really_basic_example_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
end

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
   
    global vid
    %closePreview(vid);
    img = getsnapshot(vid);
    img = ycbcr2rgb(img);
    imwrite(img,'testfile.JPEG');

    
    % Call petes function into car quality python(
    quality = python('img_quality_and_extract.py','testfile.JPEG');
    % 2 = good face and here
    % 1 = face and bad quality
    % 0 = no face seen
    
    % convert to a number
    quality = str2num(quality);
    
    if quality == 2
        % call recognition and load num_names store id in name
        
 
        answer = questdlg(sprintf('Is your username: %s', name));
        if strcmp('Yes',answer) == 1
            questdlg(sprintf('Welcome, %s', name));
            % call online training
            % close preview and program 
            
        else strcmp('Yes',answer) == 0
            prompt = {'Enter your username:'};
            actual_username = inputdlg(prompt)
            % load names info and see if they are in dic. 
            user_info = strsplit(python('load_names_info_dict.py', actual_username))'
            if strcmp(char(user_info(1)),'Not in dict')==1;
                msgbox('Not in system. Please add yourself');
            else 
                prompt = {'Enter your password:'};
                actual_password = inputdlg(prompt)
                if strcmp(char(user_info(4)),actual_password)==1;
                    msgbox(sprintf('Welcome, %s', name));
                    % call online training
                    % close preview and program
                else 
                    msgbox('invalid password: Please contact sys admin');
                    % close preview and program
                end
            end
        end
    elseif quality == 1
        msgbox('The quality of the image is too low. Please retake the photo.');
    elseif quality == 0
        msgbox('Thats not a face... please retake the photo');
    end
end


% --- Executes on button press in pushbutton3. // ie adding user
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    dict_size = (python('get_dict_size.py'));
    global vid
    global count
    %closePreview(vid);
    int i = 1;
    im_file_names = ['testfile1.JPEG';'testfile2.JPEG';'testfile3.JPEG';'testfile4.JPEG';'testfile5.JPEG'];
    while (i<6)
        img = getsnapshot(vid);
        img = ycbcr2rgb(img);
        imwrite(img,im_file_names(i,:));
        pause(1);

        quality = python('img_quality_and_extract.py',im_file_names(i,:));
        % 2 = good face and here
        % 1 = face and bad quality
        % 0 = no face seen

        % convert to a number
        quality = str2num(quality);
        if quality == 2 
        msgbox(sprintf('Picture %n taken', i));
        i = i+1;
        end

    end 
    prompt = {'Enter your username:'};
    actual_username = inputdlg(prompt)   

    prompt = {'Enter your password:'};
    actual_password = inputdlg(prompt) 
    msgbox(sprintf('Welcome, %s', actual_username));

    % call  append_dict.py
    % with something like 
    python('append_dict.py', num2str(dict_size), actual_username, actual_password);
    % call augmentation 
    % call transfer learning
end

function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double
end

% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end