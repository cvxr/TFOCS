function makeMex()
% Script to install mex files

% Change directory, so all mex files are in the mexFiles/ subdirectory
here = pwd;
cd( fullfile(tfocs_where,'mexFiles') );
if exist('OCTAVE_VERSION','builtin')
    octave = true; 
    compileFunction = @compileForOctave;
else
    octave = false;
    compileFunction = @compileForMatlab;
end


% -- Compile all the mex files --

% For the ordered L1 norm, used in prox_OL1.m:
compileFunction('proxAdaptiveL1Mex.c');

% For a faster implementation of soft-thresholding/shrinkage
compileFunction('shrink_mex.c');

if ~octave
    % For a faster-er implementation of soft-thresholding/shrinkage
    compileForMatlabWithOptions('shrink_mex2.cc');
    
    % For a faster implementation of prox_l1_and_sum
    compileForMatlabWithOptions('prox_l1_and_sum_worker_mex.cc');
end





% Change directory back to wherever we started from
cd(here);

end




% -- subroutines --
function compileForMatlab( inputFile )
    mex(inputFile)
end

function compileForMatlabWithOptions( inputFile )
    % Options are set a la mexopts.sh
    % See /usr/local/MATLAB/R2017a/bin/mexopts.sh
    try
        % for MSVC-based compilers (e.g., Windows),
        % try changing the 
        %    CFLAGS="\$CFLAGS -fopenmp" 
        % to
        %   COMPFLAGS="/openmp"
        myCCompiler = mex.getCompilerConfigurations('C','Selected');
        c_name  = lower( myCCompiler.ShortName );
        if contains( c_name, {'gcc','clang'} )
            CC_type = 1;
        elseif contains( c_name, 'msvcc' )
            CC_type = 2;
        else
            CC_type = 3;
        end
        switch CC_type
            case 1
                mex(inputFile,...
                    'CFLAGS="$CFLAGS -O2 -march=native -mtune=native -fopenmp"',...
                    'CLIBS="$CLIBS -lgomp"',...
                    'CXXFLAGS="$CXXFLAGS -O2 -march=native -mtune=native -fopenmp"',...
                    'CXXLIBS="$CXXLIBS -lgomp"')
            case 2
                % untested
                mex(inputFile,...
                    'COMPFLAGS="/openmp"' );
            case 3
                mex(inputFile )
        end
    catch er
        if strcmpi(er.identifier,'MATLAB:mex:Error') && ...
                ~isempty( strfind( er.message, 'fopenmp' ) )
            disp('Warning, openMP not found on your system, code is not multi-threaded');
            if CC_type == 1
                mex(inputFile,...
                    'CFLAGS="$CFLAGS -O2 -march=native -mtune=native"',...
                    'CLIBS="$CLIBS -lgomp"',...
                    'CXXFLAGS="$CXXFLAGS -O2 -march=native -mtune=native"',...
                    'CXXLIBS="$CXXLIBS -lgomp"')
            else
                mex(inputFile )
            end
        else
            rethrow( er );
        end
    end
end

function compileForOctave( inputFile )
if ~isunix,
    disp('warning, this will probably fail if not on linux; please edit makeMex.m yourself');
end
% To compile oct or mex files with octave, you must have the octave function "mkoctfile"
% e.g. (for ubuntu 10.04, using octave 3.2):   sudo apt-get install octave3.2-headers
%   for other versions of ubuntu and/or octave, try packages like
%       octave-pkg-dev or liboctave-dev
system(sprintf('mkoctfile --mex %s', inputFile ) );

% rm the .c ending
if length(inputFile) > 2 && inputFile(end-1:end)=='.c'
    inputFile = inputFile(1:end-2);
end
system(sprintf('rm %s.o', inputFile ) );
end


% TFOCS v1.3 by Stephen Becker, Emmanuel Candes, and Michael Grant.
% Copyright 2013 California Institute of Technology and CVX Research.
% See the file LICENSE for full license information.
