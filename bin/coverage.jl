cd(Pkg.dir("Latte"))
Pkg.add("Coverage")
using Coverage

folder = "src"
println("""Coverage.process_folder: Searching $folder for .jl files...""")
source_files = FileCoverage[]
files = readdir(folder)
for file in files
    fullfile = joinpath(folder,file)
    if isfile(fullfile)
        # Is it a Julia file?
        if splitext(fullfile)[2] == ".jl"
            push!(source_files, process_file(fullfile,folder))
        else
            println("Coverage.process_folder: Skipping $file, not a .jl file")
        end
    else isdir(fullfile)
        if file == "stdlib"
            # Skip coverage in stdlib because layer functions are never actually executed
            continue
        end
        # If it is a folder, recursively traverse
        append!(source_files, process_folder(fullfile))
    end
end

Coveralls.submit(source_files)
